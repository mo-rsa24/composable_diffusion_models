import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
import argparse  # Import argparse

from models.mlp_2d import MLP
from schedule import alpha, sigma
from utils import set_seed, save_checkpoint
from viz import plot_loss, scatter2d_labeled  # Import the new function
from dataset import get_mnist_dataloader


# --- Helper for latent space diffusion ---
def q_t_latent(x0, t, eps=None):
    """Forward diffusion for latent vectors."""
    if eps is None: eps = torch.randn_like(x0)
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    # return alpha_t * x0 + sigma_t * eps
    xt = alpha_t * x0 + sigma_t * eps
    return xt, eps


def main(args):
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    N_COMPONENTS = 2
    OUTPUT_DIR = "outputs/latent_2d"
    CHECKPOINT_DIR = "checkpoints"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 1. PCA Fitting and Data Transformation ---
    print("Fitting PCA and transforming data...")
    # Use the 'classes' argument to filter the dataset for PCA
    full_dataloader = get_mnist_dataloader(batch_size=60000, shuffle=False, classes=args.classes)
    all_images, all_labels = next(iter(full_dataloader))
    images_flat = all_images.view(all_images.size(0), -1).numpy()

    pca = PCA(n_components=N_COMPONENTS)
    latent_codes = pca.fit_transform(images_flat)

    # Save PCA components. Note: It's better to save these once for all digits.
    # We will overwrite them here, but for a full project, you might handle this differently.
    np.save(os.path.join(CHECKPOINT_DIR, "pca_mean.npy"), pca.mean_)
    np.save(os.path.join(CHECKPOINT_DIR, "pca_components.npy"), pca.components_)
    print(f"PCA mean and components saved to {CHECKPOINT_DIR}/")

    # --- Visualize Original Latent Space with Labeled Legend ---
    plot_suffix = f"_{'_'.join(map(str, args.classes))}" if args.classes else "_all"
    scatter_path = os.path.join(OUTPUT_DIR, f"latent_space_ground_truth{plot_suffix}.png")
    scatter2d_labeled(
        latent_codes,
        labels=all_labels.numpy(),
        path=scatter_path,
        title=f"MNIST Latent Space (Classes: {args.classes})"
    )
    print(f"Labeled scatter plot saved to {scatter_path}")

    # --- 2. Train Score Model (MLP) in Latent Space ---
    print(f"Starting training for classes: {args.classes}...")
    latent_dataset = TensorDataset(torch.from_numpy(latent_codes).float())
    latent_dataloader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True)

    model = MLP(num_out=N_COMPONENTS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in tqdm(range(args.epochs), desc="Training MLP"):
        for x0, in latent_dataloader:
            optimizer.zero_grad()
            x0 = x0.to(DEVICE)
            t = torch.rand(x0.shape[0], device=DEVICE) * (1.0 - 1e-3) + 1e-3
            xt, eps = q_t_latent(x0, t)
            eps_hat = model(t, xt)
            loss = ((eps - eps_hat) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    save_checkpoint(model, optimizer, args.epochs, args.model_path)
    loss_plot_path = os.path.join(OUTPUT_DIR, f"latent_loss{plot_suffix}.png")
    plot_loss(losses, loss_plot_path)
    print(f"Training complete. Model saved to {args.model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a latent diffusion model on a subset of MNIST.")

    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='List of MNIST classes to train on (e.g., 0 1 2 3 4). Default is all classes.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to save the trained model checkpoint.')

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    args = parser.parse_args()
    main(args)