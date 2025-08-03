# visualize_composition_shapes.py
import torch
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.mlp_2d import MLP
from utils import load_checkpoint, set_seed
from shapes.dataset import ShapesDataset

# --- Stable Noise Schedule ---
beta_0, beta_1 = 0.1, 20.0


def stable_log_alpha(t):
    return -0.5 * t * beta_0 - 0.25 * t.pow(2) * (beta_1 - beta_0)


def stable_alpha(t):
    return torch.exp(stable_log_alpha(t))


def stable_sigma(t):
    return torch.sqrt(1 - stable_alpha(t) ** 2)


def q_t_latent(x0, t, eps=None):
    if eps is None: eps = torch.randn_like(x0)
    alpha_t = stable_alpha(t).view(-1, 1)
    sigma_t = stable_sigma(t).view(-1, 1)
    return alpha_t * x0 + sigma_t * eps, eps


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
N_SAMPLES = 512
# (MODIFIED) Using fewer steps is fine with a good sampler like DDIM
N_STEPS = 100
W_SHAPE, W_COLOR = 1.0, 1.0

# --- Paths ---
OUTPUT_DIR = "outputs_shapes/composition_shapes"
CHECKPOINT_DIR = "checkpoints_shapes"
SHAPE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "shape_expert.pth")
COLOR_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "color_expert.pth")
PCA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pca_shapes.joblib")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Models and PCA ---
shape_model = MLP(num_out=2).to(DEVICE).eval()
color_model = MLP(num_out=2).to(DEVICE).eval()
load_checkpoint(shape_model, None, SHAPE_MODEL_PATH, DEVICE)
load_checkpoint(color_model, None, COLOR_MODEL_PATH, DEVICE)
pca = joblib.load(PCA_MODEL_PATH)

# --- 2. Load Ground Truth Data for Visualization ---
full_dataset = ShapesDataset(size=10000)
full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
all_images, all_shape_labels, all_color_labels = next(iter(full_dataloader))
images_flat = all_images.view(all_images.size(0), -1).numpy()
latent_codes = torch.from_numpy(pca.transform(images_flat)).float().to(DEVICE)

x0_up = latent_codes[all_shape_labels == 0]  # Circles
x0_down = latent_codes[all_shape_labels == 1]  # Squares

# --- 3. Run Combined Reverse Diffusion with DDIM Sampler ---
x_gen_history = {}
with torch.no_grad():
    x = torch.randn(N_SAMPLES, 2, device=DEVICE)

    # Define the time steps for DDIM
    time_steps = torch.linspace(1.0, 1e-3, N_STEPS + 1, device=DEVICE)

    for i in trange(N_STEPS, desc="Generating with DDIM Sampler"):
        t_now = time_steps[i]
        t_next = time_steps[i + 1]

        t_tensor = torch.full((N_SAMPLES,), t_now, device=DEVICE)

        # Store history at specific timesteps for plotting
        if any(np.isclose(t_now.cpu(), ts, atol=0.01) for ts in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
            x_gen_history[round(t_now.cpu().item(), 2)] = x.cpu().numpy()

        # Get combined noise prediction
        eps_hat_shape = shape_model(t_tensor, x)
        eps_hat_color = color_model(t_tensor, x)
        eps_hat_combined = (W_SHAPE * eps_hat_shape + W_COLOR * eps_hat_color) / (W_SHAPE + W_COLOR)

        # --- (MODIFIED) DDIM Update Step ---
        alpha_t_now = stable_alpha(t_tensor).view(-1, 1)
        sigma_t_now = stable_sigma(t_tensor).view(-1, 1)

        # Predict x0 based on the current xt and predicted noise
        x0_pred = (x - sigma_t_now * eps_hat_combined) / alpha_t_now
        x0_pred.clamp_(-15, 15)  # Clamp to prevent outliers, based on forward plot scale

        # Get schedule for the next step
        alpha_t_next = stable_alpha(t_next).view(-1, 1)
        sigma_t_next = stable_sigma(t_next).view(-1, 1)

        # Use the DDIM formula to deterministically step to the next xt
        # The "direction" pointing to x0 is given by eps_hat_combined
        x = alpha_t_next * x0_pred + sigma_t_next * eps_hat_combined

    x_gen_final = x
    x_gen_history[0.0] = x_gen_final.cpu().numpy()

# --- 4. Plot the Latent Space Reverse Process ---
all_plot_points = [x_gen_history[t] for t in sorted(x_gen_history.keys())]
all_plot_points.append(x0_up.cpu().numpy())
all_plot_points.append(x0_down.cpu().numpy())
all_points_np = np.concatenate(all_plot_points, axis=0)
min_val, max_val = all_points_np.min(), all_points_np.max()
plot_limit = max(abs(min_val), abs(max_val)) * 1.1

fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
plot_times = sorted(x_gen_history.keys(), reverse=True)

for i, t_val in enumerate(plot_times):
    ax = axes[i]
    t = torch.full((x0_up.shape[0],), t_val, device=DEVICE)
    xt_up, _ = q_t_latent(x0_up, t)
    t = torch.full((x0_down.shape[0],), t_val, device=DEVICE)
    xt_down, _ = q_t_latent(x0_down, t)
    xt_gen = x_gen_history[t_val]

    ax.scatter(xt_up.cpu()[:, 0], xt_up.cpu()[:, 1], alpha=0.3, label='Circles (GT)')
    ax.scatter(xt_down.cpu()[:, 0], xt_down.cpu()[:, 1], alpha=0.3, label='Squares (GT)')
    ax.scatter(xt_gen[:, 0], xt_gen[:, 1], alpha=0.5, color='purple', label='Generated')
    ax.set_title(f"t={t_val:.2f}")
    ax.grid(True)
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

axes[0].legend()
fig.suptitle("Reverse Diffusion using Combined Latent Shape/Color Models", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "latent_composition_process_shapes.png"))
plt.close()
print(f"Latent visualization saved to {os.path.join(OUTPUT_DIR, 'latent_composition_process_shapes.png')}")

# --- 5. Decode Final Latent Points back to Images ---
print("Decoding final latent points back to images...")
img_flat = pca.inverse_transform(x_gen_final.cpu().numpy())
img_reconstructed = torch.from_numpy(img_flat).view(-1, 3, 64, 64)

save_image(img_reconstructed.clamp(-1, 1), os.path.join(OUTPUT_DIR, "reconstructed_shapes.png"), nrow=16,
           normalize=True, value_range=(-1, 1))
print(f"Reconstructed images saved to {os.path.join(OUTPUT_DIR, 'reconstructed_shapes.png')}")
