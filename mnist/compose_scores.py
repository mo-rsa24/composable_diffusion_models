# compose_scores.py - Core Logic
import torch
from tqdm import trange
from models.unet_small import UNet
from schedules import dlog_alphadt, beta, sigma
from utils import load_checkpoint
from viz import save_grid

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model1_path = "checkpoints/model_0_4.pth"  # Trained on digits 0-4
model2_path = "checkpoints/model_5_9.pth"  # Trained on digits 5-9
bs = 64
n_steps = 100
dt = 1.0 / n_steps
w1, w2 = 1.0, 1.0  # Composition weights

# --- Setup ---
model1 = UNet().to(device).eval()
model2 = UNet().to(device).eval()
load_checkpoint(model1, None, model1_path, device)
load_checkpoint(model2, None, model2_path, device)

# --- Composed Sampling ---
with torch.no_grad():
    x = torch.randn(bs, 1, 28, 28, device=device)

    for i in trange(n_steps, desc=f"Composing Scores (w1={w1}, w2={w2})"):
        t = 1.0 - i * dt
        t_tensor = torch.full((bs,), t, device=device)

        eps_hat1 = model1(x, t_tensor)
        eps_hat2 = model2(x, t_tensor)

        # Weighted superposition of predicted noise
        eps_hat_combined = w1 * eps_hat1 + w2 * eps_hat2

        # SDE step using the combined noise
        drift = dlog_alphadt(t_tensor).view(-1, 1, 1, 1) * x - \
                beta(t_tensor).view(-1, 1, 1, 1) / sigma(t_tensor).view(-1, 1, 1, 1) * eps_hat_combined

        diffusion = torch.sqrt(2 * 1.0 * beta(t_tensor)).view(-1, 1, 1, 1)

        dx = -drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
        x = x + dx

save_grid(x, f"outputs/composed_samples_w1_{w1}_w2_{w2}.png")