# sample_image.py - Core Logic
import torch
from tqdm import trange
from models.unet_small import UNet
from schedules import dlog_alphadt, beta, sigma
from utils import load_checkpoint
from viz import save_grid, save_gif

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "checkpoints/image_model.pth"
bs = 64
n_steps = 100
dt = 1.0 / n_steps
xi = 1.0  # SDE noise scale

# --- Setup ---
model = UNet().to(device)
load_checkpoint(model, None, model_path, device)
model.eval()

# --- Sampling ---
frames = []
with torch.no_grad():
    x = torch.randn(bs, 1, 28, 28, device=device)

    for i in trange(n_steps, desc="Sampling"):
        t = 1.0 - i * dt
        t_tensor = torch.full((bs,), t, device=device)

        eps_hat = model(x, t_tensor)

        drift = dlog_alphadt(t_tensor).view(-1, 1, 1, 1) * x - \
                beta(t_tensor).view(-1, 1, 1, 1) / sigma(t_tensor).view(-1, 1, 1, 1) * eps_hat

        diffusion = torch.sqrt(2 * xi * beta(t_tensor)).view(-1, 1, 1, 1)

        dx = -drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
        x = x + dx

        if i % 10 == 0:
            frames.append(x[0].cpu())  # Save first image of the batch for GIF

save_grid(x, "outputs/final_samples.png")
save_gif(frames, "outputs/sampling_process.gif")