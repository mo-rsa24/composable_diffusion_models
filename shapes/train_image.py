# train_image.py - Core Logic
from pathlib import Path
import os
import torch
import torch.optim as optim
from tqdm import trange
from models.unet_small import UNet
from schedule import q_t, sigma, alpha
from utils import set_seed, save_checkpoint
from viz import save_grid, plot_loss
import argparse


# --- ⚙️ Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Shape/Color Experiment ---
    IMG_SIZE = 64
    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]
    HOLDOUT_COMBINATION = ("triangle", "blue")

    # --- 2D Toy Experiment ---
    TOY_DATA_N_SAMPLES = 512

    # --- General Training ---
    BATCH_SIZE = 128
    TIMESTEPS = 300  # Reduced for faster training/sampling
    NUM_EPOCHS = 50  # Reduced for quick demonstration
    LR = 2e-4

    # --- Directories ---
    OUTPUT_DIR = Path("diffusion_output")
    MODEL_DIR = OUTPUT_DIR / "models"
    IMAGE_DIR = OUTPUT_DIR / "images"
    TOY_2D_DIR = OUTPUT_DIR / "toy_2d"



# --- Config ---
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    epochs = 2 if args.sanity else args.epochs
    log_every_epoch= 1 if args.sanity else 20
    lr = 1e-4
    batch_size = 4 if args.sanity else args.batch_size
    sanity = True if args.sanity else False
    sanity_num_examples =  8

    # --- Create structured directories ---
    output_dir = os.path.dirname(args.model_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Setup ---
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = get_mnist_dataloader(batch_size=batch_size,  classes=args.classes, sanity=sanity, sanity_num_examples=sanity_num_examples)
    data_generator = sample_data(dataloader)
    losses = []



    # --- Training Loop ---
    for epoch in range(epochs):
        for i in trange(len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            x0 = next(data_generator).to(device)
            t = torch.rand(x0.shape[0], device=device) * (1.0 - 1e-3) + 1e-3

            xt, eps = q_t(x0, t)
            eps_hat = model(xt, t)

            loss = ((eps - eps_hat) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # --- Validation & Checkpointing ---
        if epoch % log_every_epoch == 0:
            with torch.no_grad():
                val_noise = torch.randn(64, 1, 28, 28, device=device)
                # Simple one-step denoising for a quick check
                t_val = torch.ones(64, device=device) * 0.9
                xt_val, _ = q_t(val_noise, t_val)
                eps_hat_val = model(xt_val, t_val)
                x0_hat = (xt_val - sigma(t_val).view(-1, 1, 1, 1) * eps_hat_val) / alpha(t_val).view(-1, 1, 1, 1)
                save_grid(x0_hat, f"outputs/{args.exp_name}_epoch_{epoch + 1}_val.png")
        # --- Final Checkpointing ---
    save_checkpoint(model, optimizer, epochs, args.model_path)
    loss_plot_path = os.path.join(output_dir, f"{args.exp_name}_loss_{'_'.join(map(str, args.classes))}.png")
    plot_loss(losses, loss_plot_path)
    print(f"--- Training Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a diffusion model on a subset of MNIST.")

    parser.add_argument('--classes', type=int, nargs='+', required=True,
                        help='List of MNIST classes to train on (e.g., 0 1 2 3 4).')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to save the trained model checkpoint.')

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)
