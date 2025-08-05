# train_latent_expert.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
from tqdm import tqdm
import argparse

# Assuming these modules are in the correct paths
from models.mlp_2d import MLP
from schedule_jax_faithful import alpha, sigma
# from shapes.dataset_ import ShapesDataset
from shapes.dataset_grayscale import ShapesGrayscaleDataset as ShapesDataset
from utils import set_seed, save_checkpoint


def q_t_latent(x0, t, eps=None):
    """Applies the forward diffusion process to 2D latent vectors."""
    if eps is None:
        eps = torch.randn_like(x0)
    # Reshape for broadcasting with (batch_size, 2) latent vectors
    alpha_t = alpha(t).view(-1, 1)
    sigma_t = sigma(t).view(-1, 1)
    xt = alpha_t * x0 + sigma_t * eps
    return xt, eps


def main(args):
    """Main training routine for a single-class expert model."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    # Ensure the directory for the model and PCA file exists
    checkpoint_dir = os.path.dirname(args.model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    PCA_MODEL_PATH = os.path.join(checkpoint_dir, "pca_grayscale.joblib")

    # --- 1. Load PCA and Project Data ---
    print(f"Loading PCA from {PCA_MODEL_PATH}")
    pca = joblib.load(PCA_MODEL_PATH)

    full_dataset = ShapesDataset(size=10000)
    full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
    all_images, all_shape_labels = next(iter(full_dataloader))
    images_flat = all_images.view(all_images.size(0), -1).numpy()
    latent_codes = pca.transform(images_flat)

    # --- 2. Filter Data for SINGLE Expert Training ---
    print(f"--- Training expert for SHAPE label: {args.class_label} ---")
    mask = (all_shape_labels == args.class_label)
    expert_latent_codes = latent_codes[mask]

    if len(expert_latent_codes) == 0:
        raise ValueError(f"No data found for class label {args.class_label}. Check your dataset.")

    # --- 3. Train MLP in Latent Space ---
    latent_dataset = TensorDataset(torch.from_numpy(expert_latent_codes).float())
    latent_dataloader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True)

    model = MLP(num_out=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs), desc=f"Training expert for label {args.class_label}"):
        for x0_latent, in latent_dataloader:
            optimizer.zero_grad()
            x0_latent = x0_latent.to(DEVICE)
            # Sample a random time t for each latent vector in the batch
            t = torch.rand(x0_latent.shape[0], device=DEVICE) * (1.0 - 1e-3) + 1e-3
            # Apply forward noise process
            xt_latent, eps = q_t_latent(x0_latent, t)
            # Predict noise
            eps_hat = model(t, xt_latent)
            # Calculate loss and update
            loss = ((eps - eps_hat) ** 2).mean()
            loss.backward()
            optimizer.step()

    save_checkpoint(model, optimizer, args.epochs, args.model_path)
    print(f"Training complete. Model for class {args.class_label} saved to {args.model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a latent diffusion MLP on a single shape class.")
    parser.add_argument('--class_label', type=int, required=True,
                        help="The integer label of the shape class to train on (0=circle, 1=square).")
    parser.add_argument('--model_path', type=str, required=True, help="Path to save the trained expert model.")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    main(args)
