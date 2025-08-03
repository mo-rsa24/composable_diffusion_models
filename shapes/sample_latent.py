# sample_latent_2d.py
import torch
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt

from models.mlp_2d import MLP
from schedules import dlog_alphadt, beta, sigma
from utils import load_checkpoint, set_seed
from viz import save_grid, scatter2d

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
N_SAMPLES = 64
N_COMPONENTS = 2
N_STEPS = 1000
XI = 1.0  # SDE noise scale

OUTPUT_DIR = "outputs/latent_2d"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "latent_2d_model.pth")
PCA_PATH_MEAN = os.path.join(CHECKPOINT_DIR, "pca_mean.npy")
PCA_PATH_COMPONENTS = os.path.join(CHECKPOINT_DIR, "pca_components.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Models ---
print("Loading models...")
# Load Score Network
model = MLP(num_out=N_COMPONENTS).to(DEVICE)
load_checkpoint(model, None, MODEL_PATH, DEVICE)
model.eval()

# Load PCA components
pca_mean = np.load(PCA_PATH_MEAN)
pca_components = np.load(PCA_PATH_COMPONENTS)
print("Models loaded successfully.")

# --- 2. Reverse SDE Sampling in Latent Space ---
print("Starting reverse diffusion in latent space...")
latent_frames = []
with torch.no_grad():
    # Start with random noise in the latent space
    x = torch.randn(N_SAMPLES, N_COMPONENTS, device=DEVICE)
    dt = 1.0 / N_STEPS

    for i in trange(N_STEPS, desc="Sampling Latent Codes"):
        t_val = 1.0 - i * dt
        t = torch.full((N_SAMPLES,), t_val, device=DEVICE)

        # Predict noise (score)
        eps_hat = model(t, x)

        # Euler-Maruyama step
        drift = dlog_alphadt(t).view(-1, 1) * x - \
                beta(t).view(-1, 1) / sigma(t).view(-1, 1) * eps_hat

        diffusion_scale = torch.sqrt(2 * XI * beta(t)).view(-1, 1)

        dx = -drift * dt + diffusion_scale * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
        x = x + dx

        if i % 100 == 0:
            latent_frames.append(x.cpu().numpy())

# Final generated latent codes
final_latents = x.cpu().numpy()

# --- 3. Visualize Latent Trajectories ---
plt.figure(figsize=(8, 8))
# Plot initial noise
plt.scatter(latent_frames[0][:, 0], latent_frames[0][:, 1], alpha=0.3, label='t=1.0 (Noise)')
# Plot final points
plt.scatter(final_latents[:, 0], final_latents[:, 1], alpha=0.8, label='t=0.0 (Generated)')
plt.title("Evolution of Latent Samples")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "reverse_latent_trajectory.png"))
plt.close()

# --- 4. Reconstruct Images from Latent Codes ---
print("Reconstructing images from generated latent codes...")
# Inverse transform from latent space back to pixel space
reconstructed_flat = np.dot(final_latents, pca_components) + pca_mean
reconstructed_images = torch.from_numpy(reconstructed_flat).view(N_SAMPLES, 1, 28, 28)

# Save the grid of reconstructed images
save_grid(
    reconstructed_images,
    path=os.path.join(OUTPUT_DIR, "reconstructed_samples_from_latent.png"),
    nrow=8
)
print(f"Reconstructed images saved to {OUTPUT_DIR}/")