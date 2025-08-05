# shapes/visualize_composition_grayscale.py
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
from shapes.dataset_grayscale import ShapesGrayscaleDataset
from schedule_jax_faithful import alpha, sigma, dlog_alphadt, beta

# All helper functions (q_t_latent, vector_field, get_kappa) remain the same
def q_t_latent(x0, t, eps=None):
    if eps is None: eps = torch.randn_like(x0)
    alpha_t, sigma_t = alpha(t).view(-1, 1), sigma(t).view(-1, 1)
    return alpha_t * x0 + sigma_t * eps, eps

def vector_field(model, t, x):
    x_clone = x.clone().requires_grad_(True)
    t_in = torch.full((x.shape[0],), t, device=x.device)
    eps_hat = model(t_in, x_clone)
    eps_hutch = torch.randint_like(x_clone, 0, 2, dtype=torch.float) * 2 - 1
    jvp_val = torch.autograd.grad(eps_hat, x_clone, grad_outputs=eps_hutch, create_graph=False)[0]
    return eps_hat.detach(), (jvp_val * eps_hutch).sum(dim=1)

def get_kappa(t, div_eps_hats, eps_hats, device):
    div_eps_hat_1, div_eps_hat_2 = div_eps_hats
    eps_hat_1, eps_hat_2 = eps_hats
    sigma_t = sigma(t).to(device).view(-1, 1)
    term1_num = -sigma_t * (div_eps_hat_1 - div_eps_hat_2).view(-1, 1)
    term2_num = torch.sum(eps_hat_1 * (eps_hat_1 - eps_hat_2), dim=1, keepdim=True)
    kappa_num = term1_num + term2_num
    kappa_den = torch.sum((eps_hat_1 - eps_hat_2)**2, dim=1, keepdim=True)
    kappa = kappa_num / (kappa_den + 1e-5)
    return torch.clip(kappa, -1.0, 2.0)

# --- Configuration ---
DEVICE, N_SAMPLES, N_STEPS = "cuda" if torch.cuda.is_available() else "cpu", 512, 1000
set_seed(42)

# --- Paths for Grayscale Experiment ---
CHECKPOINT_DIR = "checkpoints_shapes_grayscale"
OUTPUT_DIR = "outputs_shapes_grayscale"
CIRCLE_EXPERT_PATH = os.path.join(CHECKPOINT_DIR, "circle_expert.pth")
SQUARE_EXPERT_PATH = os.path.join(CHECKPOINT_DIR, "square_expert.pth")
PCA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pca_grayscale.joblib")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Models and PCA ---
print("Loading grayscale models and PCA...")
model_up = MLP(num_out=2).to(DEVICE).eval()
model_down = MLP(num_out=2).to(DEVICE).eval()
load_checkpoint(model_up, None, CIRCLE_EXPERT_PATH, DEVICE)
load_checkpoint(model_down, None, SQUARE_EXPERT_PATH, DEVICE)
pca = joblib.load(PCA_MODEL_PATH)

# --- 2. Load Ground Truth Data ---
print("Loading grayscale ground truth data...")
full_dataset = ShapesGrayscaleDataset(size=10000)
full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
all_images, all_shape_labels = next(iter(full_dataloader))
images_flat = all_images.view(all_images.size(0), -1).numpy()
latent_codes = torch.from_numpy(pca.transform(images_flat)).float().to(DEVICE)
x0_up = latent_codes[all_shape_labels == 0]
x0_down = latent_codes[all_shape_labels == 1]

# --- 3. Run Reverse Diffusion ---
x_gen_history, x, dt = {}, torch.randn(N_SAMPLES, 2, device=DEVICE), 1.0 / N_STEPS
for i in trange(N_STEPS, desc="Generating with Grayscale Shape Composition"):
    t_val = 1.0 - i * dt
    t = torch.full((N_SAMPLES,), t_val, device=DEVICE)
    if any(np.isclose(t_val, ts, atol=1e-3) for ts in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
        x_gen_history[round(t_val, 2)] = x.cpu().numpy()
    eps_hat_up, div_up = vector_field(model_up, t_val, x)
    eps_hat_down, div_down = vector_field(model_down, t_val, x)
    with torch.no_grad():
        kappa = get_kappa(t, (div_up, div_down), (eps_hat_up, eps_hat_down), DEVICE)
        combined_eps_hat = eps_hat_down + kappa * (eps_hat_up - eps_hat_down)
        dxdt = dlog_alphadt(t).view(-1, 1) * x + beta(t).view(-1, 1) * combined_eps_hat
        x = x - dxdt * dt
x_gen_history[0.0] = x.cpu().numpy()

# --- 4. Plot ---
print("Plotting the latent space reverse process...")
fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
plot_times = sorted(x_gen_history.keys(), reverse=True)
all_points_np = np.concatenate([x0_up.cpu().numpy(), x0_down.cpu().numpy()] + list(x_gen_history.values()))
min_val, max_val = all_points_np.min(), all_points_np.max()
plot_limit = max(abs(min_val), abs(max_val)) * 1.1
for i, t_val in enumerate(plot_times):
    ax = axes[i]
    xt_up, _ = q_t_latent(x0_up, torch.full((x0_up.shape[0],), t_val, device=DEVICE))
    xt_down, _ = q_t_latent(x0_down, torch.full((x0_down.shape[0],), t_val, device=DEVICE))
    ax.scatter(xt_up.cpu()[:, 0], xt_up.cpu()[:, 1], alpha=0.3, label='Data "Up" (Circles)')
    ax.scatter(xt_down.cpu()[:, 0], xt_down.cpu()[:, 1], alpha=0.3, label='Data "Down" (Squares)')
    ax.scatter(x_gen_history[t_val][:, 0], x_gen_history[t_val][:, 1], alpha=0.5, color='green', label='Generated (AND)')
    ax.set_title(f"t={t_val:.2f}"), ax.grid(True), ax.set_xlim(-plot_limit, plot_limit), ax.set_ylim(-plot_limit, plot_limit)
axes[0].legend(), fig.suptitle("Superposition of Grayscale Shape Experts (Circle AND Square)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "latent_composition_grayscale.png"))
plt.close()
print(f"Final plot saved to {os.path.join(OUTPUT_DIR, 'latent_composition_grayscale.png')}")

# --- 5. Decode Final Latent Points ---
print("Decoding final latent points to images...")
img_flat = pca.inverse_transform(x.cpu().numpy())
# Reshape to 1 channel for grayscale
img_reconstructed = torch.from_numpy(img_flat).view(-1, 1, 64, 64)
save_image(img_reconstructed.clamp(-1, 1), os.path.join(OUTPUT_DIR, "reconstructed_images_grayscale.png"), nrow=16, normalize=True, value_range=(-1, 1))
print(f"Reconstructed images saved to {os.path.join(OUTPUT_DIR, 'reconstructed_images_grayscale.png')}")
