# visualize_composition_latent_ito_2.py
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

# Use the single, authoritative schedule from training
# This ensures the inference dynamics match the training process.
from schedule_jax_faithful import alpha, sigma, dlog_alphadt, g2


def q_t_latent(x0, t, eps=None):
    """
    Forward diffusion for 2D latent vectors.
    """
    if eps is None:
        eps = torch.randn_like(x0)
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    return alpha_t * x0 + sigma_t * eps, eps


def vector_field(model, t, x):
    """Computes the model output (eps_hat) and its divergence for a 2D MLP."""
    x_clone = x.clone().requires_grad_(True)
    t_in = torch.full((x.shape[0],), t, device=x.device)

    eps_hat = model(t_in, x_clone)

    # Use Hutchinson's estimator for the divergence of eps_hat
    eps_hutch = torch.randn_like(x_clone)
    jvp_val = torch.autograd.grad(eps_hat, x_clone, grad_outputs=eps_hutch, create_graph=False)[0]
    divergence = (jvp_val * eps_hutch).sum(dim=1)

    return eps_hat.detach(), divergence.detach()


def get_kappa(t, div_eps_hats, eps_hats, device):
    """
    [REVISED] Calculates the weighting factor kappa for 2D latent space.

    This is a faithful PyTorch translation of the get_kappa function from the
    authors' JAX notebook (superposition_edu.py). The key is understanding the
    relationship between the model outputs:
    - This PyTorch model (`eps_hat`) predicts the noise `eps`.
    - The JAX model (`sdlogqdx`) predicts `-eps`.
    Therefore, JAX `sdlogqdx` corresponds to `-eps_hat` and JAX `divlog`
    corresponds to `-div(eps_hat)`. This function implements the JAX formula
    with the correct substitutions, which is critical for the "AND" composition.
    """
    div_eps_hat_1, div_eps_hat_2 = div_eps_hats
    eps_hat_1, eps_hat_2 = eps_hats

    sigma_t = sigma(t).to(device).view(-1, 1)

    # Numerator from JAX: sigma * (divlog_1 - divlog_2) + (sdlogdx_1 * (sdlogdx_1 - sdlogdx_2)).sum()
    # Translated to PyTorch:
    term1_num = -sigma_t * (div_eps_hat_1 - div_eps_hat_2).view(-1, 1)
    term2_num = torch.sum(eps_hat_1 * (eps_hat_1 - eps_hat_2), dim=1, keepdim=True)
    kappa_num = term1_num + term2_num

    # Denominator from JAX: ((sdlogdx_1 - sdlogdx_2)**2).sum()
    # Translated to PyTorch:
    kappa_den = torch.sum((eps_hat_1 - eps_hat_2)**2, dim=1, keepdim=True)

    kappa = kappa_num / (kappa_den + 1e-5)

    # Clipping for stability is good practice
    kappa = torch.clip(kappa, -1.0, 2.0)

    return kappa


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
print("Loading models and PCA...")
shape_model = MLP(num_out=2).to(DEVICE).eval()
color_model = MLP(num_out=2).to(DEVICE).eval()
load_checkpoint(shape_model, None, SHAPE_MODEL_PATH, DEVICE)
load_checkpoint(color_model, None, COLOR_MODEL_PATH, DEVICE)
pca = joblib.load(PCA_MODEL_PATH)

# --- 2. Load Ground Truth Data for Visualization ---
print("Loading ground truth data for visualization...")
full_dataset = ShapesDataset(size=10000)
full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
all_images, all_shape_labels, all_color_labels = next(iter(full_dataloader))
# For visualization, we'll focus on the two distributions the shape model was trained on.
shape_mask = (all_shape_labels == 0) | (all_shape_labels == 1)
images_flat_shapes = all_images[shape_mask].view(-1, 12288).numpy()
latent_codes_shapes = torch.from_numpy(pca.transform(images_flat_shapes)).float().to(DEVICE)

# 'Up' will be circles (label 0), 'Down' will be squares (label 1)
x0_up = latent_codes_shapes[all_shape_labels[shape_mask] == 0]
x0_down = latent_codes_shapes[all_shape_labels[shape_mask] == 1]

# --- 3. Run Combined Reverse Diffusion with Corrected ODE Solver ---
x_gen_history = {}
x = torch.randn(N_SAMPLES, 2, device=DEVICE)
dt = 1.0 / N_STEPS

for i in trange(N_STEPS, desc="Generating with Superposition (AND)"):
    t_val = 1.0 - i * dt
    t = torch.full((N_SAMPLES,), t_val, device=DEVICE)

    # Store history for plotting at specific intervals
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

        # Combine scores using kappa: s_comb = s_color + k*(s_shape - s_color)
        s_combined = s_color + kappa * (s_shape - s_color)

        # --- Correct Reverse ODE (Probability Flow) Update Rule ---
        dlog_alpha_dt_t = dlog_alphadt(t).view(-1, 1)
        g2_t = g2(t).view(-1, 1)

        # The probability flow ODE is: dx = [f(x,t) - 0.5*g(t)^2*s(x,t)] dt
        # where f(x,t) = dlog_alpha/dt * x
        dxdt = dlog_alpha_dt_t * x - 0.5 * g2_t * s_combined

        # Simple Euler step for the ODE: x_{t-dt} = x_t - dxdt * dt
        x = x - dxdt * dt

x_gen_final = x
x_gen_history[0.0] = x_gen_final.cpu().numpy()

# --- 4. Plot the Latent Space Reverse Process ---
print("Plotting the latent space reverse process...")
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

    ax.scatter(xt_up.cpu()[:, 0], xt_up.cpu()[:, 1], alpha=0.3, label='Data "Up" (Circles)')
    ax.scatter(xt_down.cpu()[:, 0], xt_down.cpu()[:, 1], alpha=0.3, label='Data "Down" (Squares)')
    ax.scatter(xt_gen[:, 0], xt_gen[:, 1], alpha=0.5, color='green', label='Generated (AND)')
    ax.set_title(f"t={t_val:.2f}")
    ax.grid(True)
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

axes[0].legend()
fig.suptitle("Superposition Reverse Diffusion (AND Composition)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "latent_composition_process_final.png"))
plt.close()
print(f"Latent visualization saved to {os.path.join(OUTPUT_DIR, 'latent_composition_process_final.png')}")

# --- 5. Decode Final Latent Points back to Images ---
print("Decoding final latent points back to images...")
img_flat = pca.inverse_transform(x_gen_final.cpu().numpy())
img_reconstructed = torch.from_numpy(img_flat).view(-1, 3, 64, 64)

save_image(img_reconstructed.clamp(-1, 1), os.path.join(OUTPUT_DIR, "reconstructed_images_final.png"), nrow=16,
           normalize=True, value_range=(-1, 1))
print(f"Reconstructed images saved to {os.path.join(OUTPUT_DIR, 'reconstructed_images_final.png')}")