from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from models.unet_small import UNet
from schedule import q_t, sigma, alpha
from shapes.dataset import ShapesDataset
from src.utils.tools import tiny_subset
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
    lr = args.lr
    batch_size = 4 if args.sanity else args.batch_size
    sanity = True if args.sanity else False
    sanity_num_examples =  8

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # --- Configure based on training mode ---
    if args.training_mode == 'shape':
        in_channels = 1
        dataset = ShapesDataset(mode='shape', img_size=args.img_size)
        print("--- Training SHAPE model ---")
    elif args.training_mode == 'color':
        in_channels = 3
        dataset = ShapesDataset(mode='color', img_size=args.img_size)
        print("--- Training COLOR model ---")
    else:
        raise ValueError("Invalid training mode. Choose 'shape' or 'color'.")
    if sanity:
        dataset = tiny_subset(dataset, sanity_num_examples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Instantiate U-Net with correct number of channels
    model = UNet(in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    # --- Training Loop ---
    for epoch in tqdm(range(epochs), desc=f"Training {args.training_mode} model"):
        for x0 in dataloader:
            optimizer.zero_grad()
            x0 = x0.to(device)
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
                val_noise = torch.randn(64, in_channels, args.img_size, args.img_size, device=device)
                # Simple one-step denoising for a quick check
                t_val = torch.ones(64, device=device) * 0.9
                xt_val, _ = q_t(val_noise, t_val)
                eps_hat_val = model(xt_val, t_val)
                x0_hat = (xt_val - sigma(t_val).view(-1, 1, 1, 1) * eps_hat_val) / alpha(t_val).view(-1, 1, 1, 1)
                save_grid(x0_hat, f"outputs_shapes/{args.training_mode}_epoch_{epoch + 1}_val.png")
        # --- Final Checkpointing ---
    save_checkpoint(model, optimizer, epochs, args.model_path)
    output_dir = os.path.dirname(args.model_path)
    plot_loss(losses, os.path.join(output_dir, f"loss_{args.training_mode}.png"))
    print(f"--- {args.training_mode.upper()} model saved to {args.model_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an expert diffusion model for shape or color.")
    parser.add_argument('--training_mode', type=str, required=True, choices=['shape', 'color'],
                        help="The expert to train.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to save the trained model checkpoint.")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)
