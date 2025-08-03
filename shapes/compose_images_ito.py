import torch
import os
import argparse
from tqdm import trange
from torchvision.utils import save_image
from torchvision.transforms import Grayscale
from functools import partial

# Assuming these modules are in the correct paths
from models.unet_small import UNet
# We use the schedule from the original paper for this faithful implementation
from schedule import dlog_alphadt, beta, sigma, alpha
from utils import load_checkpoint, set_seed


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]


# --- (NEW) Functions from the Itô Paper Implementation ---

def vector_field(model, t, x, y):
    """
    Computes the model output (related to the score) and its divergence.
    This is adapted for conditional models.
    """
    x_clone = x.clone().requires_grad_(True)
    t_in = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)

    # Use a partial function to fix the t and y arguments for the JVP
    model_fn = partial(model, t=t_in, y=y)

    # Random vector for Hutchinson's trace estimator
    eps = torch.randn_like(x_clone)

    # Compute JVP: J(x) * eps
    score_val, jvp_val = torch.autograd.functional.jvp(model_fn, x_clone, eps, create_graph=False)

    # Divergence is E[eps^T * J * eps]. We use a single sample.
    divergence = (jvp_val * eps).sum(dim=(1, 2, 3))

    return score_val.detach(), divergence.detach()


def get_kappa(t, divlogs, sdlogdxs, device):
    """Calculates the weighting factor kappa."""
    divlog_1, divlog_2 = divlogs
    sdlogdx_1, sdlogdx_2 = sdlogdxs

    sigma_t = sigma(t).to(device)

    # Reshape for broadcasting with image tensors
    sigma_t = sigma_t.view(-1, 1, 1, 1)

    kappa_num = sigma_t * (divlog_1 - divlog_2).view(-1, 1, 1, 1) + (sdlogdx_1 * (sdlogdx_1 - sdlogdx_2)).sum(
        dim=(1, 2, 3)).view(-1, 1, 1, 1)
    kappa_den = ((sdlogdx_1 - sdlogdx_2) ** 2).sum(dim=(1, 2, 3)).view(-1, 1, 1, 1)

    # Add a small epsilon to the denominator to avoid division by zero
    kappa = kappa_num / (kappa_den + 1e-9)
    return kappa


@torch.no_grad()
def sample_composed_ito(shape_model, color_model, shape_label, color_label, args):
    """
    Performs reverse diffusion using the Itô-based SDE solver and kappa composition.
    WARNING: This is computationally very expensive.
    """
    device = Config.DEVICE
    shape_model.eval()
    color_model.eval()

    grayscale_transform = Grayscale(num_output_channels=1)

    x = torch.randn(args.bs, 3, args.img_size, args.img_size, device=device)
    dt = 1.0 / args.n_steps

    for i in trange(args.n_steps, desc=f"Composing with Itô SDE", leave=False):
        t_val = 1.0 - i * dt

        # --- Itô Composition Step ---
        # Prepare grayscale input for the shape model
        x_gray = grayscale_transform(x)

        # Calculate scores and divergences for both models
        sdlogdx_shape, divdlog_shape = vector_field(shape_model, t_val, x_gray, shape_label)
        sdlogdx_color, divdlog_color = vector_field(color_model, t_val, x, color_label)

        # Broadcast the shape score to 3 channels
        sdlogdx_shape_rgb = sdlogdx_shape.repeat(1, 3, 1, 1)

        # Calculate kappa
        kappa = get_kappa(t_val, (divdlog_shape, divdlog_color), (sdlogdx_shape_rgb, sdlogdx_color), device)

        # --- SDE Update Step ---
        # This is the Euler-Maruyama step for the reverse SDE from the paper
        t_tensor = torch.full((args.bs,), t_val, device=device)

        # Combined score using kappa
        combined_score = sdlogdx_color + kappa * (sdlogdx_shape_rgb - sdlogdx_color)

        dxdt = dlog_alphadt(t_tensor).view(-1, 1, 1, 1) * x - beta(t_tensor).view(-1, 1, 1, 1) * combined_score

        # Simple Euler step (as in the paper's combined generation loop)
        x = x - dxdt * dt

    return x


def main(args):
    set_seed(42)
    device = Config.DEVICE
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Expert Models ---
    print("Loading expert models...")
    model_shape = UNet(in_channels=1, num_classes=len(Config.SHAPES)).to(device)
    model_color = UNet(in_channels=3, num_classes=len(Config.COLORS)).to(device)
    load_checkpoint(model_shape, None, args.shape_model_path, device)
    load_checkpoint(model_color, None, args.color_model_path, device)
    print("Models loaded successfully.")

    # --- Generate a grid of all combinations ---
    print("Generating a grid of all shape/color compositions using Itô method...")
    print("WARNING: This will be very slow due to JVP calculations for kappa.")
    all_generated_images = []

    shape_map = {name: i for i, name in enumerate(Config.SHAPES)}
    color_map = {name: i for i, name in enumerate(Config.COLORS)}

    for s_name in Config.SHAPES:
        for c_name in Config.COLORS:
            s_idx = torch.full((args.bs,), shape_map[s_name], device=device, dtype=torch.long)
            c_idx = torch.full((args.bs,), color_map[c_name], device=device, dtype=torch.long)

            composed_image = sample_composed_ito(model_shape, model_color, s_idx, c_idx, args)
            all_generated_images.append(composed_image)

    # Save the results in a grid
    grid = torch.cat(all_generated_images)
    grid_path = os.path.join(args.output_dir, "composition_grid_ito.png")
    save_image(grid, grid_path, nrow=len(Config.COLORS) * args.bs, normalize=True, value_range=(-1, 1))
    print(f"\nSaved Itô generation grid to {grid_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose shape and color diffusion models using the Itô method.")
    parser.add_argument('--shape_model_path', type=str, required=True)
    parser.add_argument('--color_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="composition_output_images_ito")
    parser.add_argument('--n_steps', type=int, default=500)  # SDE solvers often need more steps
    parser.add_argument('--bs', type=int, default=1,
                        help="Batch size (use a small value like 1 due to high computational cost).")
    parser.add_argument('--img_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
