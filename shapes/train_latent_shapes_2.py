import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
from tqdm import tqdm
import argparse

# Assuming mlp_2d.py and other modules are in correct paths
from models.mlp_2d import MLP
from schedule_2 import alpha, sigma
from shapes.dataset_ import ShapesDataset
from shapes.viz import plot_loss
from utils import set_seed, save_checkpoint

def q_t_latent(x0, t, eps=None):
    """Forward diffusion for 2D latent vectors."""
    if eps is None:
        eps = torch.randn_like(x0)
    # Reshape for broadcasting with (batch_size, 2) latent vectors
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    xt = alpha_t * x0 + sigma_t * eps
    return xt, eps
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    CHECKPOINT_DIR = "checkpoints_shapes_2"
    OUTPUT_DIR = "outputs_shapes_2/latent_shapes"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load PCA and Project Data ---
    print("Loading PCA and projecting shapes to latent space...")
    pca = joblib.load(os.path.join(CHECKPOINT_DIR, "pca_shapes.joblib"))

    full_dataset = ShapesDataset(size=10000)
    full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
    all_images, all_shape_labels, all_color_labels = next(iter(full_dataloader))

    images_flat = all_images.view(all_images.size(0), -1).numpy()
    latent_codes = pca.transform(images_flat)

    # --- 2. Filter Data for Expert Training ---
    if args.training_mode == 'shape':
        print("--- Configuring for SHAPE expert (circle vs square) ---")
        # Use latent codes for circles (label 0) and squares (label 1)
        mask = (all_shape_labels == 0) | (all_shape_labels == 1)
        expert_latent_codes = latent_codes[mask]
    elif args.training_mode == 'color':
        print("--- Configuring for COLOR expert (red vs green) ---")
        # Use latent codes for red (label 0) and green (label 1)
        mask = (all_color_labels == 0) | (all_color_labels == 1)
        expert_latent_codes = latent_codes[mask]
    else:
        raise ValueError("Invalid training mode.")

    # --- 3. Train MLP in Latent Space ---
    latent_dataset = TensorDataset(torch.from_numpy(expert_latent_codes).float())
    latent_dataloader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True)

    model = MLP(num_out=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in tqdm(range(args.epochs), desc=f"Training {args.training_mode} MLP"):
        for x0_latent, in latent_dataloader:
            optimizer.zero_grad()
            x0_latent = x0_latent.to(DEVICE)
            t = torch.rand(x0_latent.shape[0], device=DEVICE) * (1.0 - 1e-3) + 1e-3
            xt_latent, eps = q_t_latent(x0_latent, t)
            eps_hat = model(t, xt_latent)
            loss = ((eps - eps_hat) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    save_checkpoint(model, optimizer, args.epochs, args.model_path)
    plot_loss(losses, os.path.join(OUTPUT_DIR, f"loss_{args.training_mode}.png"))
    print(f"Training complete. Model saved to {args.model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a latent diffusion MLP on shape data.")
    parser.add_argument('--training_mode', type=str, required=True, choices=['shape', 'color'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    main(args)
