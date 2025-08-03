# visualize_forward_shapes.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import DataLoader

# (MODIFIED) Import alpha and sigma, as they are needed by the new helper function
from schedule import alpha, sigma
from shapes.dataset import ShapesDataset


# --- (NEW) Helper function for latent space diffusion ---
def q_t_latent(x0, t, eps=None):
    """
    Performs the forward diffusion process specifically for 2D latent vectors.
    """
    if eps is None:
        eps = torch.randn_like(x0)
    # Reshape t to (batch_size, 1) for broadcasting with latent vectors (batch_size, 2)
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    xt = alpha_t * x0 + sigma_t * eps
    return xt, eps


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs_shapes/latent_shapes"
CHECKPOINT_DIR = "checkpoints_shapes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load PCA and Project Data ---
print("Loading PCA and projecting shapes to latent space...")
pca = joblib.load(os.path.join(CHECKPOINT_DIR, "pca_shapes.joblib"))

full_dataset = ShapesDataset(size=10000)
full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
all_images, all_shape_labels, _ = next(iter(full_dataloader))

images_flat = all_images.view(all_images.size(0), -1).numpy()
latent_codes = torch.from_numpy(pca.transform(images_flat)).float().to(DEVICE)

# --- Get Latent Codes for Two Groups (e.g., Circles vs Squares) ---
x0_group1 = latent_codes[all_shape_labels == 0]  # Circles
x0_group2 = latent_codes[all_shape_labels == 1]  # Squares

# --- (MODIFIED) Pre-calculate all noised data to find plot limits ---
plot_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
all_points = []
for t_val in plot_times:
    t_group1 = torch.full((x0_group1.shape[0],), t_val, device=DEVICE)
    xt_group1, _ = q_t_latent(x0_group1, t_group1)
    all_points.append(xt_group1.cpu().numpy())

    t_group2 = torch.full((x0_group2.shape[0],), t_val, device=DEVICE)
    xt_group2, _ = q_t_latent(x0_group2, t_group2)
    all_points.append(xt_group2.cpu().numpy())

all_points_np = np.concatenate(all_points, axis=0)
min_val = all_points_np.min()
max_val = all_points_np.max()
plot_limit = max(abs(min_val), abs(max_val)) * 1.1 # Add 10% margin

# --- Plot the Forward Process ---
fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)

for i, t_val in enumerate(plot_times):
    ax = axes[i]
    # We can just use the points we already calculated
    xt_group1 = all_points[i*2]
    xt_group2 = all_points[i*2 + 1]

    ax.scatter(xt_group1[:, 0], xt_group1[:, 1], alpha=0.3, label='Circles')
    ax.scatter(xt_group2[:, 0], xt_group2[:, 1], alpha=0.3, label='Squares')

    ax.set_title(f"t={t_val:.2f}")
    ax.grid(True)
    # (MODIFIED) Use dynamic plot limits
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

axes[0].legend()
fig.suptitle("Forward Diffusion of Shapes in Latent Space", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path = os.path.join(OUTPUT_DIR, "forward_latent_process_shapes.png")
plt.savefig(save_path)
plt.close()

print(f"Forward process visualization saved to {save_path}")