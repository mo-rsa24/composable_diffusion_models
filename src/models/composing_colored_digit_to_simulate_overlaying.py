import torch
import torch.nn as nn
import math

# ==============================================================================
# 1. MODEL AND SDE DEFINITIONS (Unchanged)
# ==============================================================================
class VPSDE:
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, num_timesteps: int = 1000, device='cpu'):
        self.beta_min, self.beta_max, self.num_timesteps, self.device = beta_min, beta_max, num_timesteps, device
        self.betas = torch.linspace(beta_min, beta_max, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    # The reverse SDE requires f(x,t) and g(t)
    def f(self, x, t): return -0.5 * self.betas[t].view(-1, 1, 1, 1) * x

    def g(self, t): return torch.sqrt(self.betas[t])


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__();
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device;
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :];
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False):
        super().__init__();
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1); self.transform = nn.ConvTranspose2d(out_ch, out_ch,
                                                                                                         4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1); self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1);
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch);
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)));
        time_emb = self.relu(self.time_mlp(t));
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb;
        h = self.bnorm2(self.relu(self.conv2(h)));
        return self.transform(h)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__();
        self.time_mlp = nn.Linear(time_emb_dim, out_ch);
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1);
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch);
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)));
        time_emb = self.relu(self.time_mlp(t));
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb;
        h = self.bnorm2(self.relu(self.conv2(h)));
        return h


class ScoreModel(nn.Module):
    """The U-Net model for predicting score (noise)."""

    def __init__(self, in_channels: int = 3, time_emb_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 4),
                                      nn.ReLU(), nn.Linear(time_emb_dim * 4, time_emb_dim))
        self.initial_conv = nn.Conv2d(in_channels, 32, 3, padding=1);
        self.down1 = Block(32, 64, time_emb_dim);
        self.down2 = Block(64, 128, time_emb_dim)
        self.bot1 = Block(128, 256, time_emb_dim);
        self.up_transpose_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1);
        self.up_block_1 = ConvBlock(256, 128, time_emb_dim)
        self.up_transpose_2 = nn.ConvTranspose2d(128, 64, 4, 2, 1);
        self.up_block_2 = ConvBlock(128, 64, time_emb_dim)
        self.up_transpose_3 = nn.ConvTranspose2d(64, 32, 4, 2, 1);
        self.up_block_3 = ConvBlock(64, 32, time_emb_dim);
        self.output = nn.Conv2d(32, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t);
        x1 = self.initial_conv(x);
        x2 = self.down1(x1, t_emb);
        x3 = self.down2(x2, t_emb);
        x_bot = self.bot1(x3, t_emb)
        u1 = self.up_transpose_1(x_bot);
        u1_cat = torch.cat([u1, x3], dim=1);
        u1_out = self.up_block_1(u1_cat, t_emb)
        u2 = self.up_transpose_2(u1_out);
        u2_cat = torch.cat([u2, x2], dim=1);
        u2_out = self.up_block_2(u2_cat, t_emb)
        u3 = self.up_transpose_3(u2_out);
        u3_cat = torch.cat([u3, x1], dim=1);
        u3_out = self.up_block_3(u3_cat, t_emb)
        return self.output(u3_out)
