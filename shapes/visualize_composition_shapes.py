import torch
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.mlp_2d import MLP
from schedule import dlog_alphadt, beta, sigma, q_t
from utils import load_checkpoint, set_seed
from shapes.dataset import ShapesDataset

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
N_SAMPLES = 512
N_STEPS = 1000
W_SHAPE, W_COLOR = 1.0, 1.0  # Composition weights

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

# Define two groups for plotting (e.g., red circles vs green squares)
x0_up = latent_codes[(all_shape_labels == 0) & (all_color_labels == 0)]  # Red Circles
x0_down = latent_codes[(all_shape_labels == 1) & (all_color_labels == 1)]  # Green Squares

# --- 3. Run Combined Reverse Diffusion ---
x_gen_history = {}
with torch.no_grad():
    x = torch.randn(N_SAMPLES, 2, device=DEVICE)
    dt = 1.0 / N_STEPS

    for i in trange(N_STEPS, desc="Generating with Combined Latent Models"):
        t_val = 1.0 - i * dt
        t = torch.full((N_SAMPLES,), t_val, device=DEVICE)

        if any(np.isclose(t_val, ts, atol=1e-3) for ts in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
            x_gen_history[round(t_val, 2)] = x.cpu().numpy()

        eps_hat_shape = shape_model(t, x)
        eps_hat_color = color_model(t, x)
        eps_hat_combined = W_SHAPE * eps_hat_shape + W_COLOR * eps_hat_color

        drift = dlog_alphadt(t).view(-1, 1) * x - beta(t).view(-1, 1) / sigma(t).view(-1, 1) * eps_hat_combined
        diffusion = torch.sqrt(2 * beta(t)).view(-1, 1)
        dx = -drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
        x = x + dx

    x_gen_final = x
    x_gen_history[0.0] = x_gen_final.cpu().numpy()

# --- 4. Plot the Latent Space Reverse Process ---
fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
plot_times = sorted(x_gen_history.keys(), reverse=True)

for i, t_val in enumerate(plot_times):
    ax = axes[i]
    t = torch.full((x0_up.shape[0],), t_val, device=DEVICE)
    xt_up, _ = q_t(x0_up, t)
    t = torch.full((x0_down.shape[0],), t_val, device=DEVICE)
    xt_down, _ = q_t(x0_down, t)
    xt_gen = x_gen_history[t_val]

    ax.scatter(xt_up.cpu()[:, 0], xt_up.cpu()[:, 1], alpha=0.3, label='Red Circles')
    ax.scatter(xt_down.cpu()[:, 0], xt_down.cpu()[:, 1], alpha=0.3, label='Green Squares')
    ax.scatter(xt_gen[:, 0], xt_gen[:, 1], alpha=0.5, color='purple', label='Generated')
    ax.set_title(f"t={t_val:.2f}")
    ax.grid(True)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

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
