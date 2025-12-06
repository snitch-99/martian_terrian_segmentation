"""
segmarsvit_sine_model.py
------------------------
SegMarsViT variant using sine activations (SIREN-style) in transformer MLPs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================
# Sine activation + init
# ========================
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

def siren_init_(m):
    if isinstance(m, nn.Linear):
        in_dim = m.weight.size(1)
        bound = 1 / in_dim
        with torch.no_grad():
            m.weight.uniform_(-bound, bound)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

# ========================
# Helper convs
# ========================
def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, groups=1, act=True, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def depthwise_separable_conv(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        conv_bn_act(in_ch, in_ch, k, s, p, groups=in_ch),
        conv_bn_act(in_ch, out_ch, k=1, s=1, p=0)
    )

def patchify(x, patch_h, patch_w):
    B, C, H, W = x.shape
    ph, pw = patch_h, patch_w
    nh, nw = H // ph, W // pw
    x = x.reshape(B, C, nh, ph, nw, pw)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, nh * nw, ph * pw * C)
    return x, nh, nw

def depatchify(x, nh, nw, C, ph, pw):
    B, N, D = x.shape
    x = x.reshape(B, nh, nw, ph, pw, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, nh * ph, nw * pw)
    return x

# ========================
# Transformer with sine MLP
# ========================
class SineTransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, w0=30.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            Sine(w0=w0),
            nn.Linear(hidden, dim)
        )
        self.apply(siren_init_)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ========================
# MobileViT Block with sine transformer
# ========================
class MobileViTBlock(nn.Module):
    def __init__(self, in_ch, out_ch, transformer_dim, depth=2, num_heads=4, patch=(2,2)):
        super().__init__()
        self.local = nn.Sequential(
            conv_bn_act(in_ch, in_ch, k=3, s=1, p=1),
            conv_bn_act(in_ch, out_ch, k=1, s=1, p=0)
        )
        self.patch_h, self.patch_w = patch
        self.proj_in = nn.Linear(out_ch * self.patch_h * self.patch_w, transformer_dim)
        self.blocks = nn.ModuleList([
            SineTransformerEncoder(transformer_dim, num_heads=num_heads, w0=30.0)
            for _ in range(depth)
        ])
        self.proj_out = nn.Linear(transformer_dim, out_ch * self.patch_h * self.patch_w)
        self.fuse = conv_bn_act(out_ch, in_ch, k=1, s=1, p=0, act=False)
        self.out = conv_bn_act(in_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        y = self.local(x)
        B, C, H, W = y.shape
        ph, pw = self.patch_h, self.patch_w
        pad_h = (ph - (H % ph)) % ph
        pad_w = (pw - (W % pw)) % pw
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode='reflect')
        B, C, H2, W2 = y.shape
        tokens, nh, nw = patchify(y, ph, pw)
        z = self.proj_in(tokens)
        for blk in self.blocks:
            z = blk(z)
        z = self.proj_out(z)
        y = depatchify(z, nh, nw, C, ph, pw)
        if (H2, W2) != (H, W):
            y = y[:, :, :H, :W]
        y = self.fuse(y) + x
        y = self.out(y)
        return y

# ========================
# Encoder / Decoder / Full Model (same decoder)
# ========================
class MobileViTEncoder(nn.Module):
    def __init__(self, in_ch=3, widths=(32, 64, 96, 128), transformer_dims=(64,96,128,160),
                 depths=(1,2,3,3), heads=(2,4,4,5)):
        super().__init__()
        w1, w2, w3, w4 = widths
        t1, t2, t3, t4 = transformer_dims
        self.stem = nn.Sequential(
            conv_bn_act(in_ch, w1, 3, 2, 1),
            depthwise_separable_conv(w1, w1, 3, 1, 1),
            conv_bn_act(w1, w1, 3, 2, 1)
        )
        self.stage2_dw = depthwise_separable_conv(w1, w2, 3, 2, 1)
        self.stage2_mvit = MobileViTBlock(w2, w2, t1, depth=depths[0], num_heads=heads[0])
        self.stage3_dw = depthwise_separable_conv(w2, w3, 3, 2, 1)
        self.stage3_mvit = MobileViTBlock(w3, w3, t2, depth=depths[1], num_heads=heads[1])
        self.stage4_dw = depthwise_separable_conv(w3, w4, 3, 2, 1)
        self.stage4_mvit = MobileViTBlock(w4, w4, t3, depth=depths[2], num_heads=heads[2])
        self.stage5_mvit = MobileViTBlock(w4, w4, t4, depth=depths[3], num_heads=heads[3])
        self.out_channels = (w1, w2, w3, w4)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.stage2_mvit(self.stage2_dw(c1))
        c3 = self.stage3_mvit(self.stage3_dw(c2))
        x  = self.stage4_mvit(self.stage4_dw(c3))
        c4 = self.stage5_mvit(x)
        return c1, c2, c3, c4

def conv1x1(in_ch, out_ch): return nn.Conv2d(in_ch, out_ch, 1)

class CFF(nn.Module):
    def __init__(self, ch_low, ch_high, ch_out):
        super().__init__()
        self.low_proj  = conv_bn_act(ch_low,  ch_out, k=1, s=1, p=0)
        self.high_proj = conv_bn_act(ch_high, ch_out, k=1, s=1, p=0)
        self.mix = nn.Sequential(
            conv_bn_act(ch_out * 2, ch_out, 3, 1, 1),
            conv_bn_act(ch_out, ch_out, 3, 1, 1)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_out, max(ch_out // 4, 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch_out // 4, 16), ch_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x_low, x_high):
        x_low = self.low_proj(x_low)
        x_low = F.interpolate(x_low, size=x_high.shape[-2:], mode='bilinear', align_corners=True)
        x_high = self.high_proj(x_high)
        x = torch.cat([x_high, x_low], dim=1)
        x = self.mix(x)
        w = self.gate(x)
        return x * w + x

class CFA(nn.Module):
    def __init__(self, ch, groups=4):
        super().__init__()
        g = max(1, min(groups, ch))
        self.block = nn.Sequential(
            conv_bn_act(ch, ch, 3, 1, 1, groups=g),
            conv_bn_act(ch, ch, 1, 1, 0)
        )
        self.refine = conv_bn_act(ch, ch, 3, 1, 1)
    def forward(self, x):
        y = self.block(x)
        y = y + x
        return self.refine(y)

class ELADDecoder(nn.Module):
    def __init__(self, chs_enc, chs_dec=(160, 128, 96, 64)):
        super().__init__()
        c1, c2, c3, c4 = chs_enc
        d4, d3, d2, d1 = chs_dec
        self.lat4 = conv_bn_act(c4, d4, 1, 1, 0, act=False)
        self.lat3 = conv_bn_act(c3, d3, 1, 1, 0, act=False)
        self.lat2 = conv_bn_act(c2, d2, 1, 1, 0, act=False)
        self.lat1 = conv_bn_act(c1, d1, 1, 1, 0, act=False)
        self.cff43 = CFF(d4, d3, d3); self.cfa3 = CFA(d3)
        self.cff32 = CFF(d3, d2, d2); self.cfa2 = CFA(d2)
        self.cff21 = CFF(d2, d1, d1); self.cfa1 = CFA(d1)
        self.out_ch = d1
    def forward(self, c1, c2, c3, c4):
        x4 = self.lat4(c4)
        x3 = self.cff43(x4, self.lat3(c3)); x3 = self.cfa3(x3)
        x2 = self.cff32(x3, self.lat2(c2)); x2 = self.cfa2(x2)
        x1 = self.cff21(x2, self.lat1(c1)); x1 = self.cfa1(x1)
        return x1

class SegMarsViT_Sine(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.encoder = MobileViTEncoder(in_ch=in_channels)
        self.decoder = ELADDecoder(self.encoder.out_channels, chs_dec=(160, 128, 96, 64))
        self.pred = nn.Conv2d(self.decoder.out_ch, num_classes, 1)
    def forward(self, x):
        H, W = x.shape[-2:]
        c1, c2, c3, c4 = self.encoder(x)
        y = self.decoder(c1, c2, c3, c4)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return self.pred(y)

def build_segmarsvit(in_channels=3, num_classes=4, device=None):
    model = SegMarsViT_Sine(in_channels, num_classes)
    if device is not None:
        model = model.to(device)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
