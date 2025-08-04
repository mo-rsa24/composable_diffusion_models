# compose_images_ito.py
import torch
import os
import argparse
from tqdm import trange
from torchvision.utils import save_image
from torchvision.transforms import Grayscale

# --- [REVISED] Updated imports from the corrected schedule file ---
# We now import g2, the correct diffusion coefficient for the ODE solver,
# instead of the incorrect beta function.
from schedule_2 import sigma, dlog_alphadt, g2

from models.unet_small import UNet
from utils import load_checkpoint, set_seed


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]


# --- [REVISED] Corrected Divergence Calculation ---

def vector_field(model, t, x, y):
    """
    Computes the model output (eps_hat) and its divergence.
    This version is for models where input and output channels match (e.g., the color model).
    """
    x_clone = x.clone().requires_grad_(True)
    t_in = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)

    eps_hat = model(x_clone, t_in, y)

    # Use Hutchinson's estimator for the divergence
    eps_hutch = torch.randn_like(x_clone)
    # The `create_graph=False` is a performance optimization as we don't need gradients of the divergence.
    jvp_val = torch.autograd.grad(eps_hat, x_clone, grad_outputs=eps_hutch, create_graph=False)[0]
    divergence = (jvp_val * eps_hutch).sum(dim=(1, 2, 3))

    return eps_hat.detach(), divergence.detach()


def vector_field_grayscale(model, t, x, y, grayscale_transform):
    """
    [NEW] Correctly computes eps_hat and divergence for a model that operates on a
    grayscaled version of a 3-channel input. The divergence is computed with
    respect to the original 3-channel space, which is critical for correctness.
    """
    x_clone = x.clone().requires_grad_(True)
    t_in = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)

    # Apply grayscale transform *within* the gradient-enabled context
    x_gray = grayscale_transform(x_clone)

    # Get the 1-channel noise prediction from the shape model
    eps_hat_1_channel = model(x_gray, t_in, y)

    # Repeat the output to 3 channels to match the space of the evolving variable 'x'
    eps_hat_3_channel = eps_hat_1_channel.repeat(1, 3, 1, 1)

    # Compute divergence of the 3-channel output w.r.t the 3-channel input
    eps_hutch = torch.randn_like(x_clone)
    jvp_val = torch.autograd.grad(eps_hat_3_channel, x_clone, grad_outputs=eps_hutch, create_graph=False)[0]
    divergence = (jvp_val * eps_hutch).sum(dim=(1, 2, 3))

    return eps_hat_3_channel.detach(), divergence.detach()


def get_kappa(t, divlogs, eps_hats, device):
    """
    Calculates the composition weight kappa. This implementation correctly uses
    scores (s = -eps/sigma) as per the paper's theory.
    """
    divlog_1, divlog_2 = divlogs
    eps_hat_1, eps_hat_2 = eps_hats

    # Ensure sigma_t is on the correct device and has the right shape
    sigma_t = sigma(t).to(device).view(-1, 1, 1, 1)

    # Convert noise predictions to scores: s = -eps_hat / sigma
    s1 = -eps_hat_1 / sigma_t
    s2 = -eps_hat_2 / sigma_t

    # Divergence of score is -div(eps_hat) / sigma
    div_s1 = -divlog_1.view(-1, 1, 1, 1) / sigma_t
    div_s2 = -divlog_2.view(-1, 1, 1, 1) / sigma_t

    # This formula for kappa is faithful to the JAX implementation's intent
    kappa_num = div_s1 - div_s2 + (s1 * (s1 - s2)).sum(dim=(1, 2, 3), keepdim=True)
    kappa_den = ((s1 - s2) ** 2).sum(dim=(1, 2, 3), keepdim=True)

    # Add a small epsilon to the denominator to prevent division by zero
    kappa = kappa_num / (kappa_den + 1e-9)
    return kappa


@torch.no_grad()
def sample_composed_ito_ode(shape_model, color_model, shape_label, color_label, args):
    """
    Performs reverse diffusion using the corrected probability flow ODE solver
    and kappa-based composition.
    """
    device = Config.DEVICE
    shape_model.eval()
    color_model.eval()

    grayscale_transform = Grayscale(num_output_channels=1)

    # Start from pure Gaussian noise
    x = torch.randn(args.bs, 3, args.img_size, args.img_size, device=device)
    dt = 1.0 / args.n_steps

    for i in trange(args.n_steps, desc=f"Composing with Itô ODE", leave=False):
        t_val = 1.0 - i * dt
        t = torch.full((args.bs,), t_val, device=device)

        # Enable gradients for divergence calculation
        with torch.enable_grad():
            # [FIXED] Use the new, correct function for the shape model
            eps_hat_shape, div_shape = vector_field_grayscale(shape_model, t_val, x, shape_label, grayscale_transform)

            # The color model's channels match, so we can use the simpler function
            eps_hat_color, div_color = vector_field(color_model, t_val, x, color_label)

        # --- ODE Update Step (No gradients needed from here on) ---
        kappa = get_kappa(t, (div_shape, div_color), (eps_hat_shape, eps_hat_color), device)

        # Convert noise predictions to scores
        sigma_t = sigma(t).view(-1, 1, 1, 1)
        s_shape = -eps_hat_shape / sigma_t
        s_color = -eps_hat_color / sigma_t

        # Combine scores using the calculated kappa
        s_combined = s_color + kappa * (s_shape - s_color)

        # --- [FIXED] Correct Reverse ODE Update Rule ---
        dlog_alpha_dt_t = dlog_alphadt(t).view(-1, 1, 1, 1)

        # Use the correct diffusion coefficient g2(t) instead of the old beta(t)
        g2_t = g2(t).view(-1, 1, 1, 1)

        # This is the probability flow ODE: dx = [f(x,t) - 0.5*g(t)^2*s(x,t)] dt
        # where f(x,t) = dlog_alpha/dt * x
        dxdt = dlog_alpha_dt_t * x - 0.5 * g2_t * s_combined

        # Euler step
        x = x - dxdt * dt

    return x


def main(args):
    set_seed(42)
    device = Config.DEVICE
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading expert models...")
    model_shape = UNet(in_channels=1, num_classes=len(Config.SHAPES)).to(device)
    model_color = UNet(in_channels=3, num_classes=len(Config.COLORS)).to(device)
    load_checkpoint(model_shape, None, args.shape_model_path, device)
    load_checkpoint(model_color, None, args.color_model_path, device)
    print("Models loaded successfully.")

    print("Generating a grid of all shape/color compositions using the corrected Itô-ODE method...")
    print("WARNING: This will be slow due to JVP calculations for kappa.")
    all_generated_images = []

    shape_map = {name: i for i, name in enumerate(Config.SHAPES)}
    color_map = {name: i for i, name in enumerate(Config.COLORS)}

    for s_name in Config.SHAPES:
        for c_name in Config.COLORS:
            print(f"Generating composition for: {s_name} (shape) and {c_name} (color)")
            s_idx = torch.full((args.bs,), shape_map[s_name], device=device, dtype=torch.long)
            c_idx = torch.full((args.bs,), color_map[c_name], device=device, dtype=torch.long)

            composed_image = sample_composed_ito_ode(model_shape, model_color, s_idx, c_idx, args)
            all_generated_images.append(composed_image)

    grid = torch.cat(all_generated_images)
    grid_path = os.path.join(args.output_dir, "composition_grid_ito_ode_revised.png")
    save_image(grid, grid_path, nrow=len(Config.COLORS) * args.bs, normalize=True, value_range=(-1, 1))
    print(f"\nSaved revised Itô-ODE generation grid to {grid_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose shape and color diffusion models using the Itô-ODE method.")
    parser.add_argument('--shape_model_path', type=str, required=True)
    parser.add_argument('--color_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="composition_output_images_ito_ode")
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--bs', type=int, default=1,
                        help="Batch size (use a small value like 1 due to high computational cost).")
    parser.add_argument('--img_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
