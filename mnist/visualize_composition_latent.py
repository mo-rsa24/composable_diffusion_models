# visualize_composition_latent.py
import torch
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.mlp_2d import MLP
from schedules import dlog_alphadt, beta, sigma, alpha
from utils import load_checkpoint, set_seed
from data import get_mnist_dataloader


# --- Helper for latent space diffusion ---
def q_t_latent(x0, t, eps=None):
    if eps is None: eps = torch.randn_like(x0)
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    return alpha_t * x0 + sigma_t * eps


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
N_SAMPLES = 512
N_STEPS = 1000
W1, W2 = 1.0, 1.0  # Composition weights

# --- Paths ---
OUTPUT_DIR = "outputs/composition"
CHECKPOINT_DIR = "checkpoints"
MODEL1_PATH = os.path.join(CHECKPOINT_DIR, "latent_model_0_4.pth")
MODEL2_PATH = os.path.join(CHECKPOINT_DIR, "latent_model_5_9.pth")
PCA_PATH_MEAN = os.path.join(CHECKPOINT_DIR, "pca_mean.npy")
PCA_PATH_COMPONENTS = os.path.join(CHECKPOINT_DIR, "pca_components.npy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Models and Ground Truth Data ---
# Load Score Networks
model1 = MLP(num_out=2).to(DEVICE).eval()
model2 = MLP(num_out=2).to(DEVICE).eval()
load_checkpoint(model1, None, MODEL1_PATH, DEVICE)
load_checkpoint(model2, None, MODEL2_PATH, DEVICE)

# Load PCA components
pca_mean = np.load(PCA_PATH_MEAN)
pca_components = np.load(PCA_PATH_COMPONENTS)


# Load ground truth data and transform to latent space
def get_latent_codes_for_classes(classes):
    dataloader = get_mnist_dataloader(batch_size=10000, classes=classes)
    images, _ = next(iter(dataloader))
    images_flat = images.view(images.size(0), -1).numpy()
    latent_codes = np.dot(images_flat - pca_mean, pca_components.T)
    return torch.from_numpy(latent_codes).float().to(DEVICE)


x0_up = get_latent_codes_for_classes([0, 1, 2, 3, 4])
x0_down = get_latent_codes_for_classes([5, 6, 7, 8, 9])

# --- 2. Run Combined Reverse Diffusion ---
x_gen_history = {}
with torch.no_grad():
    x = torch.randn(N_SAMPLES, 2, device=DEVICE)
    dt = 1.0 / N_STEPS

    for i in trange(N_STEPS, desc="Generating with Combined Latent Models"):
        t_val = 1.0 - i * dt
        t = torch.full((N_SAMPLES,), t_val, device=DEVICE)

        # Store history at specific timesteps for plotting
        if any(np.isclose(t_val, ts) for ts in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
            x_gen_history[round(t_val, 2)] = x.cpu().numpy()

        # Score superposition
        eps_hat1 = model1(t, x)
        eps_hat2 = model2(t, x)
        eps_hat_combined = W1 * eps_hat1 + W2 * eps_hat2

        # SDE step
        drift = dlog_alphadt(t).view(-1, 1) * x - beta(t).view(-1, 1) / sigma(t).view(-1, 1) * eps_hat_combined
        diffusion = torch.sqrt(2 * 1.0 * beta(t)).view(-1, 1)
        dx = -drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
        x = x + dx

    # Ensure t=0.0 is captured
    x_gen_history[0.0] = x.cpu().numpy()

# --- 3. Plot the Results ---
fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
plot_times = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

for i, t_val in enumerate(plot_times):
    ax = axes[i]
    t = torch.full((x0_up.shape[0],), t_val, device=DEVICE)

    # Get noised ground truth data
    xt_up = q_t_latent(x0_up, t).cpu().numpy()
    xt_down = q_t_latent(x0_down, t).cpu().numpy()

    # Get generated data from history
    xt_gen = x_gen_history[t_val]

    ax.scatter(xt_up[:, 0], xt_up[:, 1], alpha=0.3, label='noise_data_up (0-4)')
    ax.scatter(xt_down[:, 0], xt_down[:, 1], alpha=0.3, label='noise_data_down (5-9)')
    ax.scatter(xt_gen[:, 0], xt_gen[:, 1], alpha=0.5, color='green', label='gen_data')

    ax.set_title(f"t={t_val:.2f}")
    ax.grid(True)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

axes[0].legend()
fig.suptitle("Reverse Diffusion using Combined Latent Models", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "latent_composition_process.png"))
plt.close()

print(f"Visualization saved to {os.path.join(OUTPUT_DIR, 'latent_composition_process.png')}")