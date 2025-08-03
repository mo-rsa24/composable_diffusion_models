# compose_scores.py - Core Logic
import torch
import os
from tqdm import trange
from models.unet_small import UNet
from schedule import dlog_alphadt, beta, sigma
from utils import load_checkpoint
from viz import save_grid
import argparse

# --- Config ---
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    dt = 1.0 / args.n_steps

    # --- Setup ---
    print(f"Loading expert model 1 from: {args.model1_path}")
    model1 = UNet().to(device).eval()
    load_checkpoint(model1, None, args.model1_path, device)

    print(f"Loading expert model 2 from: {args.model2_path}")
    model2 = UNet().to(device).eval()
    load_checkpoint(model2, None, args.model2_path, device)
    # --- Composed Sampling ---
    with torch.no_grad():
        x = torch.randn(args.bs, 1, 28, 28, device=device)
        desc = f"Composing (w1={args.w1}, w2={args.w2})"
        for i in trange(args.n_steps, desc=desc):
            t_val = 1.0 - i * dt
            t = torch.full((args.bs,), t_val, device=device)

            eps_hat1 = model1(x, t)
            eps_hat2 = model2(x, t)

            # Weighted superposition of predicted noise
            eps_hat_combined = args.w1 * eps_hat1 + args.w2 * eps_hat2

            # SDE step
            drift = dlog_alphadt(t).view(-1, 1, 1, 1) * x - \
                    beta(t).view(-1, 1, 1, 1) / sigma(t).view(-1, 1, 1, 1) * eps_hat_combined

            diffusion = torch.sqrt(2 * args.xi * beta(t)).view(-1, 1, 1, 1)

            dx = -drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
            x = x + dx

    save_grid(x, args.output_file)
    print(f"Composed samples saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose two expert diffusion models.")

    parser.add_argument('--model1_path', type=str, required=True, help='Path to the first expert model.')
    parser.add_argument('--model2_path', type=str, required=True, help='Path to the second expert model.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output grid image.')

    parser.add_argument('--w1', type=float, default=1.0, help='Weight for the first model.')
    parser.add_argument('--w2', type=float, default=1.0, help='Weight for the second model.')

    parser.add_argument('--bs', type=int, default=64, help='Batch size for sampling.')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of sampling steps.')
    parser.add_argument('--xi', type=float, default=1.0, help='SDE noise scale.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)