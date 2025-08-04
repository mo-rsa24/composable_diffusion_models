from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Grayscale, GaussianBlur
from tqdm import tqdm
from models.unet_small import UNet
from schedule import q_t, sigma, alpha
from shapes.dataset import ShapesDataset
from src.utils.tools import tiny_subset
from utils import set_seed, save_checkpoint
from viz import save_grid_ as save_grid, plot_loss
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


@torch.no_grad()
def sample_full_ddim(model, num_samples, num_classes, device, img_size, in_channels, timesteps=100):
    """
    Performs a full reverse diffusion process using a DDIM-like deterministic sampler
    to generate high-quality images for validation.
    """
    model.eval()

    # Create a batch of labels that cycles through all available classes
    val_labels = torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)
    val_labels = val_labels[:num_samples]

    # Start from pure noise
    x = torch.randn(num_samples, in_channels, img_size, img_size, device=device)

    # Define the time steps to sample at
    time_steps = torch.linspace(1.0, 1e-3, timesteps + 1, device=device)

    for i in tqdm(range(timesteps), desc="Full DDIM Sampling", leave=False):
        t_now = time_steps[i]
        t_next = time_steps[i + 1]

        t_tensor = torch.full((num_samples,), t_now, device=device)

        # Predict the noise (or score)
        eps_hat = model(x, t_tensor, val_labels)

        # Predict x0 using the formula from the one-step validation
        alpha_t = alpha(t_tensor).view(-1, 1, 1, 1)
        sigma_t = sigma(t_tensor).view(-1, 1, 1, 1)
        x0_pred = (x - sigma_t * eps_hat) / alpha_t
        x0_pred.clamp_(-1, 1)

        # Use the DDIM formula to step to the next time step
        alpha_t_next = alpha(t_next).view(-1, 1, 1, 1)
        sigma_t_next = sigma(t_next).view(-1, 1, 1, 1)

        # The "direction" pointing to x0 is eps_hat.
        # We use it to step from our predicted x0 to the next noise level.
        x = alpha_t_next * x0_pred + sigma_t_next * eps_hat

    model.train()
    return x

# --- Config ---
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    epochs = 2 if args.sanity else args.epochs
    log_every_epoch= 1 if args.sanity else 100
    val_log_every_epoch= 1 if args.sanity else 50
    lr = args.lr
    batch_size = 4 if args.sanity else args.batch_size
    sanity = True if args.sanity else False
    sanity_num_examples =  8

    model_dir = Path(os.path.dirname(args.model_path))
    output_dir = Path("final_output_shapes")
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Configure based on training mode ---
    if args.training_mode == 'shape':
        print("--- Configuring for SHAPE model training ---")
        in_channels = 1
        num_classes = len(Config.SHAPES)
        transform = Grayscale(num_output_channels=1)
    elif args.training_mode == 'color':
        print("--- Configuring for COLOR model training ---")
        in_channels = 3
        num_classes = len(Config.COLORS)
        # Preprocessing for color model: blur to obscure shape details
        transform = GaussianBlur(kernel_size=9, sigma=5.0)
    else:
        raise ValueError("Invalid training mode. Choose 'shape' or 'color'.")

    # --- 2. Data Loading ---
    # The dataset now returns (image_tensor, shape_label, color_label)
    dataset = ShapesDataset(size=5000, img_size=args.img_size)
    if sanity:
        dataset = tiny_subset(dataset, sanity_num_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate U-Net with correct number of channels
    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    # --- Training Loop ---
    print(f"--- Starting training for {args.training_mode.upper()} model ---")
    for epoch in tqdm(range(epochs), desc=f"Training {args.training_mode} model"):
        for batch in dataloader:
            optimizer.zero_grad()
            x0, shape_labels, color_labels = batch
            x0 = x0.to(device)

            processed_images = transform(x0)
            # Select the correct labels for conditioning
            if args.training_mode == 'shape':
                labels = shape_labels.to(device)
            else:  # 'color'
                labels = color_labels.to(device)

            t = torch.rand(processed_images.shape[0], device=device) * (1.0 - 1e-3) + 1e-3
            xt, eps = q_t(processed_images, t)
            eps_hat = model(xt, t, labels)
            loss = ((eps - eps_hat) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (epoch + 1) % val_log_every_epoch == 0:
                print(f"\nEpoch {epoch + 1}, Loss: {loss.item():.4f}")
                print("Running one-step validation sampling...")
                model.eval()
                with torch.no_grad():
                    num_val_samples = 16
                    val_labels = torch.arange(num_classes, device=device).repeat(num_val_samples // num_classes + 1)[
                                 :num_val_samples]
                    val_noise = torch.randn(num_val_samples, in_channels, args.img_size, args.img_size, device=device)
                    t_val = torch.ones(num_val_samples, device=device) * 0.99
                    xt_val, _ = q_t(val_noise, t_val)
                    eps_hat_val = model(xt_val, t_val, val_labels)
                    alpha_t = alpha(t_val).view(-1, 1, 1, 1)
                    sigma_t = sigma(t_val).view(-1, 1, 1, 1)
                    x0_hat = (xt_val - sigma_t * eps_hat_val) / alpha_t
                    training_mode_dir = output_dir / args.training_mode
                    training_mode_dir.mkdir(parents=True, exist_ok=True)
                    save_grid(x0_hat, str(training_mode_dir /  f"epoch_{epoch + 1}_onestep_val.png"),
                              in_channels)
                model.train()

            if (epoch + 1) % log_every_epoch == 0:
                print("Running high-quality full validation sampling...")
                full_samples = sample_full_ddim(model, 16, num_classes, device, args.img_size, in_channels)
                training_mode_dir = output_dir / args.training_mode
                training_mode_dir.mkdir(parents=True, exist_ok=True)
                save_grid(full_samples, str(training_mode_dir / f"epoch_{epoch + 1}_full_val.png"),
                          in_channels)

    # --- 6. Final Checkpointing and Logging ---
    save_checkpoint(model, optimizer, args.epochs, args.model_path)
    plot_loss(losses, str(model_dir / f"loss_{args.training_mode}.png"))
    print(f"--- {args.training_mode.upper()} model saved to {args.model_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an expert diffusion model for shape or color.")
    parser.add_argument('--training_mode', type=str, required=True, choices=['shape', 'color'],
                        help="The expert to train.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to save the trained model checkpoint.")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)
