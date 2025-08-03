# models/unet_small.py
import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # Downsampling path
        self.down1 = ResBlock(base_dim, base_dim, time_emb_dim)
        self.down2 = ResBlock(base_dim, base_dim * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = ResBlock(base_dim * 2, base_dim * 4, time_emb_dim)

        # Upsampling path
        self.up1 = ResBlock(base_dim * 4 + base_dim * 2, base_dim * 2, time_emb_dim)
        self.up2 = ResBlock(base_dim * 2 + base_dim, base_dim, time_emb_dim)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Output
        self.out_conv = nn.Conv2d(base_dim, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)

        b1 = self.bot1(self.pool(d2), t_emb)

        u1 = self.unpool(b1)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1, t_emb)

        u2 = self.unpool(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2, t_emb)

        return self.out_conv(u2)