"""
voice_filter_model.py — StudioEnhanceNet (Transformer U-Net)

Fixes in this version:
  • Proper CRM head initialization INSIDE the model class
  • Clean quantization function (no syntax errors)
  • Stable identity initialization (important for SI-SNR sanity checks)
  • center=True consistent with ISTFT reconstruction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SR, N_FFT, HOP, F_BINS = 16000, 512, 128, 257
CRM_SAT = 10.0


# ─────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.GELU(),

            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=(2, 1), padding=1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            c_in, c_out, 3,
            stride=(2, 1),
            padding=1,
            output_padding=(1, 0)
        )
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.gelu(self.bn(self.up(x)))


class TransformerBottleneck(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_heads=4, n_layers=4):
        super().__init__()

        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, in_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        b, c, f, t = x.shape

        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.proj_in(x)
        x = self.tr(x)
        x = self.proj_out(x)

        return x.reshape(b, t, c, f).permute(0, 2, 3, 1)


# ─────────────────────────────────────────────────────────────
class StudioEnhanceNet(nn.Module):
    def __init__(self, base_ch=32, bottleneck_dim=256):
        super().__init__()

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Encoder
        self.in_block = ConvBlock(2, c1)
        self.down1 = Down(c1, c2)
        self.enc2 = ConvBlock(c2, c2)

        self.down2 = Down(c2, c3)
        self.enc3 = ConvBlock(c3, c3)

        self.down3 = Down(c3, c4)

        # Bottleneck
        self.bottleneck = TransformerBottleneck(
            in_dim=c4 * 33,
            hidden_dim=bottleneck_dim
        )
        self.bot_block = ConvBlock(c4, c4)

        # Decoder
        self.up3 = Up(c4, c3)
        self.dec3 = ConvBlock(c3 * 2, c3)

        self.up2 = Up(c3, c2)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = Up(c2, c1)
        self.dec1 = ConvBlock(c1 * 2, c1)

        # ── CRM head ──
        self.crm_head = nn.Conv2d(c1, 2, 1)

        # IMPORTANT: proper identity initialization
        self._init_crm_head()

    def _init_crm_head(self):
        nn.init.zeros_(self.crm_head.weight)

        with torch.no_grad():
            self.crm_head.bias.zero_()
            self.crm_head.bias[0] = 1.0  # real = identity
            self.crm_head.bias[1] = 0.0  # imag = identity

    def forward(self, x_real, x_imag):
        x = torch.stack([x_real, x_imag], dim=1)  # (B, 2, F, T)

        f_orig = x.shape[2]
        pad_f = (8 - f_orig % 8) % 8
        if pad_f:
            x = F.pad(x, (0, 0, 0, pad_f))

        e1 = self.in_block(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.down3(e3)

        b = self.bottleneck(b)
        b = self.bot_block(b)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        if pad_f:
            d1 = d1[:, :, :f_orig, :]

        # ── Complex Ratio Mask ──
        crm = self.crm_head(d1)
        crm_r, crm_i = crm[:, 0], crm[:, 1]

        crm_r = torch.tanh(crm_r / CRM_SAT) * CRM_SAT
        crm_i = torch.tanh(crm_i / CRM_SAT) * CRM_SAT

        y_real = crm_r * x_real - crm_i * x_imag
        y_imag = crm_r * x_imag + crm_i * x_real

        return y_real, y_imag


# ─────────────────────────────────────────────────────────────
def reconstruct_audio(real, imag, length=None):
    spec = torch.complex(real, imag)

    win = torch.hann_window(N_FFT, device=spec.device)

    return torch.istft(
        spec,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=win,
        center=True,
        length=length
    )


# ─────────────────────────────────────────────────────────────
def quantise_model(model):
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )