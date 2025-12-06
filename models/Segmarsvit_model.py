"""
segmarsvit_model.py
-------------------
MobileViT encoder + ELAD decoder (“SegMarsViT”) for 4-class semantic segmentation.

- Encoder: MobileViT-style hybrid (local conv + global self-attention over patches)
- Decoder: ELAD (Cross-Scale Feature Fusion + Compact Feature Aggregation)
- Output: per-pixel logits

This module is self-contained (PyTorch only). Import `build_segmarsvit` to construct the model.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helpers
# -----------------------------
def conv_bn_act(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1,
                groups: int = 1, act: bool = True, bn: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def depthwise_separable_conv(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        conv_bn_act(in_ch, in_ch, k, s, p, groups=in_ch),  # depthwise
        conv_bn_act(in_ch, out_ch, k=1, s=1, p=0)          # pointwise
    )


def patchify(x: torch.Tensor, patch_h: int, patch_w: int) -> Tuple[torch.Tensor, int, int]:
    """
    Convert a feature map (B,C,H,W) into patch tokens (B, N, P*C),
    where P = patch_h*patch_w and N = (H/patch_h)*(W/patch_w).
    """
    B, C, H, W = x.shape
    assert H % patch_h == 0 and W % patch_w == 0, "H and W must be multiples of patch size"
    ph, pw = patch_h, patch_w
    nh, nw = H // ph, W // pw
    x = x.reshape(B, C, nh, ph, nw, pw)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, nh * nw, ph * pw * C)  # (B, N, P*C)
    return x, nh, nw


def depatchify(x: torch.Tensor, nh: int, nw: int, C: int, ph: int, pw: int) -> torch.Tensor:
    """
    Inverse of patchify. Convert tokens (B, N, P*C) back to feature map (B, C, nh*ph, nw*pw).
    """
    B, N, D = x.shape
    assert N == nh * nw and D == ph * pw * C
    x = x.reshape(B, nh, nw, ph, pw, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, nh * ph, nw * pw)
    return x


# -----------------------------
# Transformer (ViT-like) block used inside MobileViT
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# MobileViT block: local conv -> global transformer on patches -> fusion
# -----------------------------
class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        transformer_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        patch: Tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        self.local = nn.Sequential(
            conv_bn_act(in_ch, in_ch, k=3, s=1, p=1),
            conv_bn_act(in_ch, out_ch, k=1, s=1, p=0),
        )
        self.patch_h, self.patch_w = patch
        self.proj_in = nn.Linear(out_ch * self.patch_h * self.patch_w, transformer_dim)
        self.blocks = nn.ModuleList([TransformerEncoder(transformer_dim, num_heads=num_heads) for _ in range(depth)])
        self.proj_out = nn.Linear(transformer_dim, out_ch * self.patch_h * self.patch_w)
        self.fuse = conv_bn_act(out_ch, in_ch, k=1, s=1, p=0, act=False)  # residual alignment (no act)
        self.out = conv_bn_act(in_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local(x)  # (B, C_out, H, W)
        B, C, H, W = y.shape
        ph, pw = self.patch_h, self.patch_w

        # pad to multiples of patch size if needed
        pad_h = (ph - (H % ph)) % ph
        pad_w = (pw - (W % pw)) % pw
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode='reflect')
        _, C2, H2, W2 = y.shape

        tokens, nh, nw = patchify(y, ph, pw)  # (B, N, P*C)
        z = self.proj_in(tokens)               # (B, N, D)
        for blk in self.blocks:
            z = blk(z)
        z = self.proj_out(z)                   # (B, N, P*C)
        y = depatchify(z, nh, nw, C2, ph, pw)  # (B, C_out, H2, W2)

        # remove padding if any
        if (H2, W2) != (H, W):
            y = y[:, :, :H, :W]

        y = self.fuse(y) + x  # residual to input channels
        y = self.out(y)
        return y


# -----------------------------
# MobileViT-style Encoder (lightweight)
# Returns multi-scale features at strides 4, 8, 16, 32
# -----------------------------
class MobileViTEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: Tuple[int, int, int, int] = (32, 64, 96, 128),
        transformer_dims: Tuple[int, int, int, int] = (64, 96, 128, 160),
        depths: Tuple[int, int, int, int] = (1, 2, 3, 3),
        heads: Tuple[int, int, int, int] = (2, 4, 4, 5),
    ) -> None:
        super().__init__()
        w1, w2, w3, w4 = widths
        t1, t2, t3, t4 = transformer_dims

        # Stem: stride 2 -> stride 4
        self.stem = nn.Sequential(
            conv_bn_act(in_ch, w1, k=3, s=2, p=1),           # /2
            depthwise_separable_conv(w1, w1, k=3, s=1, p=1),
            conv_bn_act(w1, w1, k=3, s=2, p=1),              # /4
        )
        # Stage 2 (/8)
        self.stage2_dw = depthwise_separable_conv(w1, w2, k=3, s=2, p=1)
        self.stage2_mvit = MobileViTBlock(w2, w2, transformer_dim=t1, depth=depths[0], num_heads=heads[0], patch=(2, 2))

        # Stage 3 (/16)
        self.stage3_dw = depthwise_separable_conv(w2, w3, k=3, s=2, p=1)
        self.stage3_mvit = MobileViTBlock(w3, w3, transformer_dim=t2, depth=depths[1], num_heads=heads[1], patch=(2, 2))

        # Stage 4 (/32)
        self.stage4_dw = depthwise_separable_conv(w3, w4, k=3, s=2, p=1)
        self.stage4_mvit = MobileViTBlock(w4, w4, transformer_dim=t3, depth=depths[2], num_heads=heads[2], patch=(2, 2))

        # Final refinement at /32
        self.stage5_mvit = MobileViTBlock(w4, w4, transformer_dim=t4, depth=depths[3], num_heads=heads[3], patch=(2, 2))

        # Expose encoder channel sizes for the decoder
        self.out_channels = (w1, w2, w3, w4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stem(x)                              # /4
        c2 = self.stage2_mvit(self.stage2_dw(c1))      # /8
        c3 = self.stage3_mvit(self.stage3_dw(c2))      # /16
        x  = self.stage4_mvit(self.stage4_dw(c3))      # /32
        c4 = self.stage5_mvit(x)                       # /32 refined
        return c1, c2, c3, c4


# -----------------------------
# ELAD: Cross-Scale Feature Fusion (CFF) + Compact Feature Aggregation (CFA)
# -----------------------------
class CFF(nn.Module):
    """
    Fuse high-res (skip) with low-res (decoder) efficiently:
    1) Align channels by 1x1
    2) Upsample low-res to match spatial
    3) Concatenate and apply lightweight mixing + gating
    """
    def __init__(self, ch_low: int, ch_high: int, ch_out: int) -> None:
        super().__init__()
        self.low_proj  = conv_bn_act(ch_low,  ch_out, k=1, s=1, p=0)
        self.high_proj = conv_bn_act(ch_high, ch_out, k=1, s=1, p=0)
        self.mix = nn.Sequential(
            conv_bn_act(ch_out * 2, ch_out, k=3, s=1, p=1),
            conv_bn_act(ch_out, ch_out, k=3, s=1, p=1),
        )
        # Channel-attention gate (SE-like)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_out, max(ch_out // 4, 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch_out // 4, 16), ch_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        x_low = self.low_proj(x_low)
        x_low = F.interpolate(x_low, size=x_high.shape[-2:], mode='bilinear', align_corners=True)
        x_high = self.high_proj(x_high)
        x = torch.cat([x_high, x_low], dim=1)
        x = self.mix(x)
        w = self.gate(x)
        return x * w + x  # gated residual


class CFA(nn.Module):
    """
    Compact Feature Aggregation:
    1) Grouped conv to reduce compute
    2) 1x1 conv for channel mixing (MLP-like)
    3) Residual refinement
    """
    def __init__(self, ch: int, groups: int = 4) -> None:
        super().__init__()
        g = max(1, min(groups, ch))
        self.block = nn.Sequential(
            conv_bn_act(ch, ch, k=3, s=1, p=1, groups=g),
            conv_bn_act(ch, ch, k=1, s=1, p=0),
        )
        self.refine = conv_bn_act(ch, ch, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        y = y + x
        return self.refine(y)


class ELADDecoder(nn.Module):
    """
    Progressive fusion from /32 up to /4 using CFF + CFA.
    Expect encoder outputs: c1(/4), c2(/8), c3(/16), c4(/32)
    """
    def __init__(self, chs_enc: Tuple[int, int, int, int], chs_dec: Tuple[int, int, int, int] = (160, 128, 96, 64)) -> None:
        super().__init__()
        c1, c2, c3, c4 = chs_enc
        d4, d3, d2, d1 = chs_dec

        # project encoder channels to decoder dims
        self.lat4 = conv_bn_act(c4, d4, k=1, s=1, p=0, act=False)
        self.lat3 = conv_bn_act(c3, d3, k=1, s=1, p=0, act=False)
        self.lat2 = conv_bn_act(c2, d2, k=1, s=1, p=0, act=False)
        self.lat1 = conv_bn_act(c1, d1, k=1, s=1, p=0, act=False)

        self.cff43 = CFF(d4, d3, d3)
        self.cfa3  = CFA(d3)

        self.cff32 = CFF(d3, d2, d2)
        self.cfa2  = CFA(d2)

        self.cff21 = CFF(d2, d1, d1)
        self.cfa1  = CFA(d1)

        self.out_ch = d1

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        x4 = self.lat4(c4)                         # /32
        x3 = self.cff43(x4, self.lat3(c3))         # fuse to /16
        x3 = self.cfa3(x3)

        x2 = self.cff32(x3, self.lat2(c2))         # fuse to /8
        x2 = self.cfa2(x2)

        x1 = self.cff21(x2, self.lat1(c1))         # fuse to /4
        x1 = self.cfa1(x1)
        return x1                                   # /4


# -----------------------------
# Full SegMarsViT
# -----------------------------
class SegMarsViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        widths: Tuple[int, int, int, int] = (32, 64, 96, 128),
        transformer_dims: Tuple[int, int, int, int] = (64, 96, 128, 160),
        depths: Tuple[int, int, int, int] = (1, 2, 3, 3),
        heads: Tuple[int, int, int, int] = (2, 4, 4, 5),
    ) -> None:
        super().__init__()
        self.encoder = MobileViTEncoder(
            in_ch=in_channels,
            widths=widths,
            transformer_dims=transformer_dims,
            depths=depths,
            heads=heads,
        )
        self.decoder = ELADDecoder(self.encoder.out_channels, chs_dec=(160, 128, 96, 64))
        self.pred = nn.Conv2d(self.decoder.out_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        c1, c2, c3, c4 = self.encoder(x)
        y = self.decoder(c1, c2, c3, c4)                 # /4 feature
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        logits = self.pred(y)
        return logits


# -----------------------------
# Public API
# -----------------------------
def build_segmarsvit(in_channels: int = 3, num_classes: int = 4, device: Optional[torch.device] = None) -> SegMarsViT:
    model = SegMarsViT(in_channels=in_channels, num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "SegMarsViT",
    "build_segmarsvit",
    "count_parameters",
]
