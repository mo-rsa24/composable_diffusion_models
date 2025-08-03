# train_latent_2d.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.mlp_2d import MLP
from schedule import q_t as q_t_image, alpha, sigma
from utils import set_seed, save_checkpoint
from viz import plot_loss, scatter2d
from dataset import get_mnist_dataloader

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 512
N_COMPONENTS = 2
OUTPUT_DIR = "outputs/latent_2d"
CHECKPOINT_DIR = "checkpoints"
PCA_PATH_MEAN = os.path.join(CHECKPOINT_DIR, "pca_mean.npy")
PCA_PATH_COMPONENTS = os.path.join(CHECKPOINT_DIR, "pca_components.npy")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "latent_2d_model.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --- Helper for latent space diffusion ---
def q_t_latent(x0, t, eps=None):
    """Forward diffusion for latent vectors."""
    if eps is None:
        eps = torch.randn_like(x0)

    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)

    xt = alpha_t * x0 + sigma_t * eps
    return xt, eps


# --- 1. PCA Fitting and Data Transformation ---
print("Fitting PCA and transforming data...")
# Load the full dataset to fit PCA
full_dataloader = get_mnist_dataloader(batch_size=60000, shuffle=False)
all_images, all_labels = next(iter(full_dataloader))
images_flat = all_images.view(all_images.size(0), -1).numpy()

# Fit PCA and transform the data
pca = PCA(n_components=N_COMPONENTS)
latent_codes = pca.fit_transform(images_flat)

# Save PCA components for reconstruction
np.save(PCA_PATH_MEAN, pca.mean_)
np.save(PCA_PATH_COMPONENTS, pca.components_)
print(f"PCA mean and components saved to {CHECKPOINT_DIR}/")

# --- Visualize Original Latent Space ---
scatter2d(
    latent_codes,
    labels=all_labels.numpy(),
    path=os.path.join(OUTPUT_DIR, "latent_space_ground_truth.png"),
    title="MNIST Latent Space (PCA)"
)

# --- 2. Train Score Model (MLP) in Latent Space ---
print("Starting training in latent space...")
latent_dataset = TensorDataset(torch.from_numpy(latent_codes).float())
latent_dataloader = DataLoader(latent_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MLP(num_out=N_COMPONENTS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
losses = []

for epoch in range(EPOCHS):
    for i, (x0,) in enumerate(tqdm(latent_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
        optimizer.zero_grad()

        x0 = x0.to(DEVICE)
        t = torch.rand(x0.shape[0], device=DEVICE) * (1.0 - 1e-3) + 1e-3

        xt, eps = q_t_latent(x0, t)
        eps_hat = model(t, xt)

        loss = ((eps - eps_hat) ** 2).mean()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            losses.append(loss.item())

save_checkpoint(model, optimizer, epoch, MODEL_PATH)
plot_loss(losses, os.path.join(OUTPUT_DIR, "latent_loss.png"))
print(f"Training complete. Model saved to {MODEL_PATH}")

# --- 3. Visualize Forward Process in Latent Space ---
print("Visualizing forward process in latent space...")
# Get a batch of ground truth latent codes
x0_sample = next(iter(latent_dataloader))[0].to(DEVICE)

plt.figure(figsize=(18, 4))
for i, t_val in enumerate([0.01, 0.25, 0.5, 0.75, 1.0]):
    plt.subplot(1, 5, i + 1)
    xt_sample, _ = q_t_latent(x0_sample, torch.full((x0_sample.shape[0],), t_val, device=DEVICE))
    plt.scatter(xt_sample[:, 0].cpu().numpy(), xt_sample[:, 1].cpu().numpy(), alpha=0.4)
    plt.title(f"t = {t_val:.2f}")
    plt.axis('square')
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
plt.suptitle("Forward Diffusion in 2D Latent Space")
plt.savefig(os.path.join(OUTPUT_DIR, "forward_latent_process.png"))
plt.close()
print(f"Forward process visualization saved.")