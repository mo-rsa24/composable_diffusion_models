# visualize_composition_latent_ito.py
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

# --- [FIXED] Use the single, authoritative schedule from training ---
# This ensures the inference dynamics match the training process.
# We import g2, the correct diffusion coefficient for the ODE solver.
from schedule_2 import alpha, sigma, dlog_alphadt, g2


def q_t_latent(x0, t, eps=None):
    """
    [FIXED] Forward diffusion for 2D latent vectors.
    This function now correctly uses the imported schedule functions.
    """
    if eps is None: eps = torch.randn_like(x0)
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    return alpha_t * x0 + sigma_t * eps, eps


# --- Functions from the Itô Paper for Latent Space ---
def vector_field(model, t, x):
    """Computes the model output (eps_hat) and its divergence for a 2D MLP."""
    x_clone = x.clone().requires_grad_(True)
    t_in = torch.full((x.shape[0],), t, device=x.device)

    eps_hat = model(t_in, x_clone)

    eps_hutch = torch.randn_like(x_clone)
    jvp_val = torch.autograd.grad(eps_hat, x_clone, grad_outputs=eps_hutch, create_graph=False)[0]
    divergence = (jvp_val * eps_hutch).sum(dim=1)

    return eps_hat.detach(), divergence.detach()


def get_kappa(t, divlogs, eps_hats, device):
    """
    Calculates the weighting factor kappa for 2D latent space.
    This function was already correctly implemented.
    """
    divlog_1, divlog_2 = divlogs
    eps_hat_1, eps_hat_2 = eps_hats

    sigma_t = sigma(t).to(device)

    s1 = -eps_hat_1 / sigma_t.view(-1, 1)
    s2 = -eps_hat_2 / sigma_t.view(-1, 1)

    div_s1 = -divlog_1 / sigma_t
    div_s2 = -divlog_2 / sigma_t

    kappa_num = div_s1 - div_s2 + (s1 * (s1 - s2)).sum(dim=1)
    kappa_den = ((s1 - s2) ** 2).sum(dim=1)

    kappa = kappa_num / (kappa_den + 1e-9)
    return kappa.view(-1, 1)


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
_, all_shape_labels, all_color_labels = next(iter(full_dataloader))
# For visualization, we'll focus on the two distributions the shape model was trained on.
shape_mask = (all_shape_labels == 0) | (all_shape_labels == 1)
images_flat_shapes = _[shape_mask].view(-1, 12288).numpy()
latent_codes_shapes = torch.from_numpy(pca.transform(images_flat_shapes)).float().to(DEVICE)

x0_up = latent_codes_shapes[all_shape_labels[shape_mask] == 0]  # Circles
x0_down = latent_codes_shapes[all_shape_labels[shape_mask] == 1]  # Squares

# --- 3. Run Combined Reverse Diffusion with Corrected ODE Solver ---
x_gen_history = {}
x = torch.randn(N_SAMPLES, 2, device=DEVICE)
dt = 1.0 / N_STEPS

for i in trange(N_STEPS, desc="Generating with Corrected Reverse ODE"):
    t_val = 1.0 - i * dt
    t = torch.full((N_SAMPLES,), t_val, device=DEVICE)

    # Store history for plotting
    if any(np.isclose(t_val, ts, atol=1e-3) for ts in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
        x_gen_history[round(t_val, 2)] = x.cpu().numpy()

    # --- Itô Composition Step (Requires gradients for divergence) ---
    eps_hat_shape, div_shape = vector_field(shape_model, t_val, x)
    eps_hat_color, div_color = vector_field(color_model, t_val, x)

    # --- ODE Update Step (No gradients needed from here on) ---
    with torch.no_grad():
        kappa = get_kappa(t, (div_shape, div_color), (eps_hat_shape, eps_hat_color), DEVICE)

        # Convert noise predictions to scores
        sigma_t_unsq = sigma(t).view(-1, 1)
        s_shape = -eps_hat_shape / sigma_t_unsq
        s_color = -eps_hat_color / sigma_t_unsq

        # Combine scores using kappa
        s_combined = s_color + kappa * (s_shape - s_color)

        # --- [FIXED] Correct Reverse ODE (Probability Flow) Update Rule ---
        dlog_alpha_dt_t = dlog_alphadt(t).view(-1, 1)
        g2_t = g2(t).view(-1, 1)  # Use the correct g2 coefficient

        # The probability flow ODE is: dx = [f(x,t) - 0.5*g(t)^2*s(x,t)] dt
        # where f(x,t) = dlog_alpha/dt * x
        dxdt = dlog_alpha_dt_t * x - 0.5 * g2_t * s_combined

        # Simple Euler step for the ODE: x_{t-dt} = x_t - dxdt * dt
        x = x - dxdt * dt

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
    # Get ground truth noised data for visualization
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
fig.suptitle("Corrected Reverse Diffusion using Combined Latent Models (Itô-ODE)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "latent_composition_process_corrected.png"))
plt.close()
print(f"Latent visualization saved to {os.path.join(OUTPUT_DIR, 'latent_composition_process_corrected.png')}")

# --- 5. Decode Final Latent Points back to Images ---
print("Decoding final latent points back to images...")
img_flat = pca.inverse_transform(x_gen_final.cpu().numpy())
img_reconstructed = torch.from_numpy(img_flat).view(-1, 3, 64, 64)

save_image(img_reconstructed.clamp(-1, 1), os.path.join(OUTPUT_DIR, "reconstructed_images_corrected.png"), nrow=16,
           normalize=True, value_range=(-1, 1))
print(f"Reconstructed images saved to {os.path.join(OUTPUT_DIR, 'reconstructed_images_corrected.png')}")
