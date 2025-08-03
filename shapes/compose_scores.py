import torch
import os
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms import Grayscale
# Assuming these modules are in the correct paths
from models.unet_small import UNet
from schedule import alpha, sigma
from utils import load_checkpoint, set_seed


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]


@torch.no_grad()
def sample_composed_ddim(shape_model, color_model, shape_label, color_label, args):
    """
    Performs a full reverse diffusion process using a DDIM sampler to generate
    a high-quality image by composing a shape and color model.
    """
    device = Config.DEVICE
    shape_model.eval()
    color_model.eval()

    # Start from pure noise
    x = torch.randn(args.bs, 3, args.img_size, args.img_size, device=device)

    grayscale_transform = Grayscale(num_output_channels=1)

    # Define the time steps for DDIM
    time_steps = torch.linspace(1.0, 1e-3, args.n_steps + 1, device=device)

    for i in tqdm(range(args.n_steps),
                  desc=f"Composing {Config.SHAPES[shape_label[0]]} + {Config.COLORS[color_label[0]]}", leave=False):
        t_now = time_steps[i]
        t_next = time_steps[i + 1]

        t_tensor = torch.full((args.bs,), t_now, device=device)

        # --- Model Prediction ---
        # Prepare grayscale input for the shape model
        x_gray = grayscale_transform(x)

        # Predict noise from each expert, providing the correct labels
        eps_hat_shape = shape_model(x_gray, t_tensor, shape_label)
        eps_hat_color = color_model(x, t_tensor, color_label)

        # Broadcast the grayscale shape noise to 3 channels
        eps_hat_shape_rgb = eps_hat_shape.repeat(1, 3, 1, 1)

        # --- Score Composition ---
        # Combine the noise predictions using a weighted average
        eps_hat_combined = (args.w_shape * eps_hat_shape_rgb + args.w_color * eps_hat_color) / (
                    args.w_shape + args.w_color)

        # --- DDIM Update Step ---
        alpha_t_now = alpha(t_tensor).view(-1, 1, 1, 1)
        sigma_t_now = sigma(t_tensor).view(-1, 1, 1, 1)

        # Predict x0
        x0_pred = (x - sigma_t_now * eps_hat_combined) / alpha_t_now
        x0_pred.clamp_(-1, 1)

        # Get schedule for the next step
        alpha_t_next = alpha(t_next).view(-1, 1, 1, 1)
        sigma_t_next = sigma(t_next).view(-1, 1, 1, 1)

        # Step to the next xt
        x = alpha_t_next * x0_pred + sigma_t_next * eps_hat_combined

    return x


def main(args):
    set_seed(42)
    device = Config.DEVICE
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Expert Models ---
    print("Loading expert models...")
    # Shape model is 1-channel and conditioned on number of shapes
    model_shape = UNet(in_channels=1, num_classes=len(Config.SHAPES)).to(device)
    # Color model is 3-channel and conditioned on number of colors
    model_color = UNet(in_channels=3, num_classes=len(Config.COLORS)).to(device)

    load_checkpoint(model_shape, None, args.shape_model_path, device)
    load_checkpoint(model_color, None, args.color_model_path, device)
    print("Models loaded successfully.")

    # --- Generate a grid of all combinations ---
    print("Generating a grid of all shape/color compositions...")
    all_generated_images = []

    shape_map = {name: i for i, name in enumerate(Config.SHAPES)}
    color_map = {name: i for i, name in enumerate(Config.COLORS)}

    for s_name in Config.SHAPES:
        for c_name in Config.COLORS:
            # Create labels for the desired shape and color
            s_idx = torch.full((args.bs,), shape_map[s_name], device=device, dtype=torch.long)
            c_idx = torch.full((args.bs,), color_map[c_name], device=device, dtype=torch.long)

            # Generate the composed image
            composed_image = sample_composed_ddim(model_shape, model_color, s_idx, c_idx, args)
            all_generated_images.append(composed_image)

    # Save the results in a grid
    grid = torch.cat(all_generated_images)
    grid_path = os.path.join(args.output_dir, "composition_grid.png")
    save_image(grid, grid_path, nrow=len(Config.COLORS) * args.bs, normalize=True, value_range=(-1, 1))
    print(f"\nSaved generation grid to {grid_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose shape and color diffusion models in pixel space.")
    parser.add_argument('--shape_model_path', type=str, required=True, help="Path to the trained shape expert model.")
    parser.add_argument('--color_model_path', type=str, required=True, help="Path to the trained color expert model.")
    parser.add_argument('--output_dir', type=str, default="composition_output_images")
    parser.add_argument('--n_steps', type=int, default=200, help="Number of steps for DDIM sampling.")
    parser.add_argument('--bs', type=int, default=4,
                        help="Batch size for generation (number of examples per combination).")
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--w_shape', type=float, default=1.0, help="Weight for the shape model's score.")
    parser.add_argument('--w_color', type=float, default=1.0, help="Weight for the color model's score.")
    args = parser.parse_args()
    main(args)
