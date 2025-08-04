# visualize_composition_latent_avg.py
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
from shapes.dataset_ import ShapesDataset

# --- Use the single, authoritative schedule from training ---
# This ensures the inference dynamics match the training process.
from schedule_2 import alpha, sigma, dlog_alphadt, g2

def q_t_latent(x0, t, eps=None):
    """
    Forward diffusion for 2D latent vectors.
    This function correctly uses the imported schedule functions.
    """
    if eps is None: eps = torch.randn_like(x0)
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    return alpha_t * x0 + sigma_t * eps, eps


# --- [NEW] Simplified function to get model output ---
# Since we use simple averaging, we no longer need the divergence calculation.
# This makes the generation loop faster.
def get_eps_hat(model, t, x):
    """Computes the model output (eps_hat) for a 2D MLP without divergence."""
    with torch.no_grad():
        t_in = torch.full((x.shape[0],), t, device=x.device)
        eps_hat = model(t_in, x)
    return eps_hat


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
N_SAMPLES = 512
N_STEPS = 1000

# --- Paths ---
OUTPUT_DIR = "outputs_shapes_2/composition_shapes"
CHECKPOINT_DIR = "checkpoints_shapes_2"
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
all_images, all_shape_labels, _ = next(iter(full_dataloader))
shape_mask = (all_shape_labels == 0) | (all_shape_labels == 1)
images_flat_shapes = all_images[shape_mask].view(-1, 12288).numpy()
latent_codes_shapes = torch.from_numpy(pca.transform(images_flat_shapes)).float().to(DEVICE)

x0_up = latent_codes_shapes[all_shape_labels[shape_mask] == 0]    # Circles
x0_down = latent_codes_shapes[all_shape_labels[shape_mask] == 1] # Squares

# --- 3. Run Combined Reverse Diffusion with Simple Averaging ---
x_gen_history = {}
x = torch.randn(N_SAMPLES, 2, device=DEVICE)
dt = 1.0 / N_STEPS

for i in trange(N_STEPS, desc="Generating with Simple Averaging ODE"):
    t_val = 1.0 - i * dt
    t = torch.full((N_SAMPLES,), t_val, device=DEVICE)

    if any(np.isclose(t_val, ts, atol=1e-3) for ts in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
        x_gen_history[round(t_val, 2)] = x.cpu().numpy()

    # Get noise predictions from both models
    eps_hat_shape = get_eps_hat(shape_model, t_val, x)
    eps_hat_color = get_eps_hat(color_model, t_val, x)

    with torch.no_grad():
        # --- [MODIFIED] Use a fixed kappa for simple averaging ---
        kappa = 0.5

        # Convert noise predictions to scores
        sigma_t_unsq = sigma(t).view(-1, 1)
        s_shape = -eps_hat_shape / sigma_t_unsq
        s_color = -eps_hat_color / sigma_t_unsq

        # Combine scores. With kappa=0.5, this is a simple average: (s_color + s_shape) / 2
        s_combined = s_color + kappa * (s_shape - s_color)

        # --- Use the Correct Reverse ODE (Probability Flow) Update Rule ---
        dlog_alpha_dt_t = dlog_alphadt(t).view(-1, 1)
        g2_t = g2(t).view(-1, 1)

        dxdt = dlog_alpha_dt_t * x - 0.5 * g2_t * s_combined

        # Euler step
        x = x - dxdt * dt

x_gen_final = x
x_gen_history[0.0] = x_gen_final.cpu().numpy()

# --- 4. Plot the Latent Space Reverse Process ---
# This plotting logic remains the same.
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
    t_tensor_up = torch.full((x0_up.shape[0],), t_val, device=DEVICE)
    xt_up, _ = q_t_latent(x0_up, t_tensor_up)
    t_tensor_down = torch.full((x0_down.shape[0],), t_val, device=DEVICE)
    xt_down, _ = q_t_latent(x0_down, t_tensor_down)
    xt_gen = x_gen_history[t_val]

    ax.scatter(xt_up.cpu()[:, 0], xt_up.cpu()[:, 1], alpha=0.3, label='Data "Up" (GT)')
    ax.scatter(xt_down.cpu()[:, 0], xt_down.cpu()[:, 1], alpha=0.3, label='Data "Down" (GT)')
    ax.scatter(xt_gen[:, 0], xt_gen[:, 1], alpha=0.5, color='green', label='Generated')
    ax.set_title(f"t={t_val:.2f}")
    ax.grid(True)
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

axes[0].legend()
fig.suptitle("Reverse Diffusion using Simple Averaging of Latent Models", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "latent_composition_process_simple_avg.png"))
plt.close()
print(f"Latent visualization saved to {os.path.join(OUTPUT_DIR, 'latent_composition_process_simple_avg.png')}")

# --- 5. Decode Final Latent Points back to Images ---
print("Decoding final latent points back to images...")
img_flat = pca.inverse_transform(x_gen_final.cpu().numpy())
img_reconstructed = torch.from_numpy(img_flat).view(-1, 3, 64, 64)

save_image(img_reconstructed.clamp(-1, 1), os.path.join(OUTPUT_DIR, "reconstructed_images_simple_avg.png"), nrow=16,
           normalize=True, value_range=(-1, 1))
print(f"Reconstructed images saved to {os.path.join(OUTPUT_DIR, 'reconstructed_images_simple_avg.png')}")
