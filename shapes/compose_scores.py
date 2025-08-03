# compose_shapes.py
import torch
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_grayscale

from models.unet_small import UNet
from schedule import dlog_alphadt, beta, sigma
from utils import load_checkpoint, set_seed
from viz import save_grid, save_gif


# --- Visualization Helper for Score Fields ---
def plot_score_field(score_map, image, path, title):
    """Visualizes a score vector field on top of an image."""
    score_map = score_map.squeeze(0).cpu().numpy()
    image = (image.squeeze(0).cpu().permute(1, 2, 0) + 1) / 2.0

    # Subsample the score map for a cleaner plot
    step = 4
    X, Y = torch.meshgrid(torch.arange(0, score_map.shape[2], step), torch.arange(0, score_map.shape[1], step))
    U = score_map[0, Y, X]  # x-component of score
    V = score_map[1, Y, X]  # y-component of score

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.quiver(X, Y, U, V, color='red', scale=50, headwidth=4)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.savefig(path)
    plt.close()


def main(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    dt = 1.0 / args.n_steps

    # --- Load Expert Models ---
    model_shape = UNet(in_channels=1).to(device).eval()
    model_color = UNet(in_channels=3).to(device).eval()
    load_checkpoint(model_shape, None, args.shape_model_path, device)
    load_checkpoint(model_color, None, args.color_model_path, device)

    # --- Reverse Diffusion Loop (Main Function) ---
    @torch.no_grad()
    def sample(mode='combined', w_shape=1.0, w_color=1.0):
        x = torch.randn(args.bs, 3, args.img_size, args.img_size, device=device)
        frames = []

        for i in tqdm(range(args.n_steps), desc=f"Sampling ({mode})"):
            t_val = 1.0 - i * dt
            t = torch.full((args.bs,), t_val, device=device)

            # Prepare inputs for models
            x_gray = to_grayscale(x, num_output_channels=1)

            # Predict noise from each expert
            eps_hat_shape = model_shape(x_gray, t)
            eps_hat_color = model_color(x, t)

            # --- Score Fusion ---
            # Broadcast shape noise to 3 channels to match color noise
            eps_hat_shape_rgb = eps_hat_shape.repeat(1, 3, 1, 1)

            if mode == 'combined':
                eps_hat_combined = w_shape * eps_hat_shape_rgb + w_color * eps_hat_color
            elif mode == 'shape_only':
                eps_hat_combined = eps_hat_shape_rgb
            elif mode == 'color_only':
                eps_hat_combined = eps_hat_color

            # --- SDE Step ---
            drift = dlog_alphadt(t).view(-1, 1, 1, 1) * x - beta(t).view(-1, 1, 1, 1) / sigma(t).view(-1, 1, 1,
                                                                                                      1) * eps_hat_combined
            diffusion = torch.sqrt(2 * 1.0 * beta(t)).view(-1, 1, 1, 1)
            dx = -drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
            x = x + dx

            if (i % 50 == 0) or (i == args.n_steps - 1):
                frames.append(x.cpu())

        return frames

    # --- Generate and Save Visualizations ---
    # 1. Individual model results
    print("Generating for shape model only...")
    shape_frames = sample('shape_only')
    save_grid(shape_frames[-1], os.path.join(args.output_dir, "final_shape_only.png"))
    save_gif([grid[0] for grid in shape_frames], os.path.join(args.output_dir, "reverse_shape.gif"))

    print("Generating for color model only...")
    color_frames = sample('color_only')
    save_grid(color_frames[-1], os.path.join(args.output_dir, "final_color_only.png"))
    save_gif([grid[0] for grid in color_frames], os.path.join(args.output_dir, "reverse_color.gif"))

    # 2. Combined model results
    print("Generating for combined model...")
    combined_frames = sample('combined', w_shape=args.w_shape, w_color=args.w_color)
    save_grid(combined_frames[-1], os.path.join(args.output_dir, "final_combined.png"))
    save_gif([grid[0] for grid in combined_frames], os.path.join(args.output_dir, "reverse_combined.gif"))

    # 3. Score Vector Field Visualization at a specific timestep
    print("Visualizing score fields at t=0.5...")
    with torch.no_grad():
        noisy_image = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        t_viz = torch.tensor([0.5], device=device)

        # Shape score
        eps_shape = model_shape(to_grayscale(noisy_image, 1), t_viz)
        plot_score_field(eps_shape, to_grayscale(noisy_image, 1),
                         os.path.join(args.output_dir, "score_field_shape.png"), "Shape Score Field")

        # Color score
        eps_color = model_color(noisy_image, t_viz)
        plot_score_field(eps_color[:, 0:2, :, :], noisy_image, os.path.join(args.output_dir, "score_field_color.png"),
                         "Color Score Field")  # Plot only 2D components

        # Combined score
        eps_combined = args.w_shape * eps_shape.repeat(1, 3, 1, 1) + args.w_color * eps_color
        plot_score_field(eps_combined[:, 0:2, :, :], noisy_image,
                         os.path.join(args.output_dir, "score_field_combined.png"), "Combined Score Field")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose shape and color diffusion models.")
    parser.add_argument('--shape_model_path', type=str, required=True)
    parser.add_argument('--color_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="composition_output")
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--w_shape', type=float, default=1.0)
    parser.add_argument('--w_color', type=float, default=1.0)
    args = parser.parse_args()
    main(args)