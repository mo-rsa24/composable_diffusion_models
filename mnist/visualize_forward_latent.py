# visualize_forward_latent.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from schedule import alpha, sigma
from dataset import get_mnist_dataloader


# --- Helper for latent space diffusion ---
def q_t_latent(x0, t, eps=None):
    if eps is None: eps = torch.randn_like(x0)
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    return alpha_t * x0 + sigma_t * eps


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs/latent_2d"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Universal PCA ---
PCA_PATH_MEAN = os.path.join(CHECKPOINT_DIR, "pca_mean.npy")
PCA_PATH_COMPONENTS = os.path.join(CHECKPOINT_DIR, "pca_components.npy")
pca_mean = np.load(PCA_PATH_MEAN)
pca_components = np.load(PCA_PATH_COMPONENTS)


# --- Load and Transform Ground Truth Data ---
def get_latent_codes_for_classes(classes):
    dataloader = get_mnist_dataloader(batch_size=10000, classes=classes)
    images, _ = next(iter(dataloader))
    images_flat = images.view(images.size(0), -1).numpy()
    latent_codes = np.dot(images_flat - pca_mean, pca_components.T)
    return torch.from_numpy(latent_codes).float().to(DEVICE)


x0_up = get_latent_codes_for_classes([0, 1, 2, 3, 4])
x0_down = get_latent_codes_for_classes([5, 6, 7, 8, 9])

# --- Plot the Forward Process ---
fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
plot_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for i, t_val in enumerate(plot_times):
    ax = axes[i]
    t_up = torch.full((x0_up.shape[0],), t_val, device=DEVICE)
    t_down = torch.full((x0_down.shape[0],), t_val, device=DEVICE)

    # Get noised ground truth data
    xt_up = q_t_latent(x0_up, t_up).cpu().numpy()
    xt_down = q_t_latent(x0_down, t_down).cpu().numpy()

    ax.scatter(xt_up[:, 0], xt_up[:, 1], alpha=0.3, label='data_up (0-4)')
    ax.scatter(xt_down[:, 0], xt_down[:, 1], alpha=0.3, label='data_down (5-9)')

    ax.set_title(f"t={t_val:.2f}")
    ax.grid(True)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

axes[0].legend()
fig.suptitle("Forward Diffusion in Latent Space", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "forward_latent_process.png"))
plt.close()

print(f"Forward process visualization saved to {os.path.join(OUTPUT_DIR, 'forward_latent_process.png')}")