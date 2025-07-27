"""
Intent
Compose two separately trained diffusion models—one on shape only,
 one on color only—to generate colored digits that neither model saw during training.

Hypothesis
If the grayscale “shape” model has learned pure digit morphology and the “color” model has learned color textures,
then fusing their score estimates at sampling time will produce a digit in the desired color,
 even for digit–color pairs unseen during training.
"""
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torchvision
from box import Box
import math
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from torchvision.utils import save_image


model_A_name, model_B_name = "Object", "Color"

# ==============================================================================
# 1. MODEL AND SDE DEFINITIONS (Unchanged)
# ==============================================================================
# (VPSDE, SinusoidalPosEmb, ColoredMNISTScoreModel, etc. are included here)
class VPSDE:
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, num_timesteps: int = 1000, device='cpu'):
        self.beta_min, self.beta_max, self.num_timesteps, self.device = beta_min, beta_max, num_timesteps, device
        self.betas = torch.linspace(beta_min, beta_max, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch,
                                                                                                         4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h


class ColoredMNISTScoreModel(nn.Module):
    def __init__(self, in_channels: int = 3, time_emb_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 4),
                                      nn.ReLU(), nn.Linear(time_emb_dim * 4, time_emb_dim))
        self.initial_conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.down1 = Block(32, 64, time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim)
        self.bot1 = Block(128, 256, time_emb_dim)
        self.up_transpose_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up_block_1 = ConvBlock(256, 128, time_emb_dim)
        self.up_transpose_2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up_block_2 = ConvBlock(128, 64, time_emb_dim)
        self.up_transpose_3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up_block_3 = ConvBlock(64, 32, time_emb_dim)
        self.output = nn.Conv2d(32, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x1 = self.initial_conv(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x_bot = self.bot1(x3, t_emb)
        u1 = self.up_transpose_1(x_bot)
        u1_cat = torch.cat([u1, x3], dim=1)
        u1_out = self.up_block_1(u1_cat, t_emb)
        u2 = self.up_transpose_2(u1_out)
        u2_cat = torch.cat([u2, x2], dim=1)
        u2_out = self.up_block_2(u2_cat, t_emb)
        u3 = self.up_transpose_3(u2_out)
        u3_cat = torch.cat([u3, x1], dim=1)
        u3_out = self.up_block_3(u3_cat, t_emb)
        return self.output(u3_out)


# ==============================================================================
# 2. CUSTOM DATASETS
# ==============================================================================
class GrayscaleMNIST(Dataset):
    """MNIST dataset filtered by digit, but kept as 3-channel grayscale."""

    def __init__(self, image_size, target_digits):
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        self.indices = [i for i, (_, label) in enumerate(mnist) if label in target_digits]
        self.mnist_dataset = mnist

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[self.indices[idx]]
        image_tensor = self.transform(image)
        # Repeat the single channel to get a 3-channel grayscale image and normalize to [-1, 1]
        final_image = (image_tensor.repeat(3, 1, 1) * 2) - 1
        return final_image, label


class SimpleShapesDataset(Dataset):
    """A synthetic dataset of simple, solid-colored shapes."""

    def __init__(self, image_size, num_samples, shape_color):
        self.image_size = image_size
        self.num_samples = num_samples
        self.shape_color = torch.tensor(shape_color).float().view(3, 1, 1)  # [C, H, W]

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        # Create a black canvas
        image = torch.zeros(3, self.image_size, self.image_size)
        # Draw a square of random size and position
        square_size = np.random.randint(self.image_size // 4, self.image_size // 2)
        x_start = np.random.randint(0, self.image_size - square_size)
        y_start = np.random.randint(0, self.image_size - square_size)
        image[:, y_start:y_start + square_size, x_start:x_start + square_size] = self.shape_color
        # Normalize to [-1, 1]
        return (image * 2) - 1, 0  # Label is unused

def get_dataset(name, image_size, **kwargs):
    if name.lower() == 'grayscalemnist':
        return GrayscaleMNIST(image_size, **kwargs)
    elif name.lower() == 'simpleshapesdataset':
        return SimpleShapesDataset(image_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
# ==============================================================================
# 3. TRAINING AND SAMPLING (Mostly Unchanged)
# ==============================================================================

class CheckpointManager:
    """A simple checkpoint manager that saves to a structured directory."""

    def __init__(self, base_dir, exp_name, run_id):
        self.base_dir = Path(base_dir) / exp_name / run_id

    def get_path(self, type='checkpoints'):
        path = self.base_dir  / type
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, model, model_name, epoch=None):
        filename = f"{model_name}_final.pth" if epoch is None else f"{model_name}_epoch_{epoch}.pth"
        save_path = self.get_path('checkpoints') / filename
        torch.save(model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")

    def load(self, model, model_name, device, epoch=None):
        filename = f"{model_name}_final.pth" if epoch is None else f"{model_name}_epoch_{epoch}.pth"
        load_path = self.get_path('checkpoints') / filename
        if not load_path.exists(): raise FileNotFoundError(f"Checkpoint {load_path} not found.")
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded model checkpoint from {load_path}")
        return model

def train(cfg, model, sde, train_loader, device, model_name, ckpt_mgr, results_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.params.lr)
    print(f"--- Starting Training for {model_name} ---")
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}"):
            optimizer.zero_grad()
            x0 = images.to(device)
            t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            sqrt_alpha_bar_t = sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
            predicted_noise = model(xt, t.float())
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()
        if epoch % cfg.training.log_every_epoch == 0:
            model.eval()
            sampler = SuperDiffSampler(sde)
            shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)
            print("Generating samples from Model A (Content)...")
            samples = sampler.sample_single_model(model, cfg.sampling.batch_size, shape, device)

            path_a: Path = results_dir / model_name / f"epoch_{epoch}"
            path_a.mkdir(parents=True, exist_ok=True)

            print(f"Saved Model A samples to {path_a}")
            for (i, img) in enumerate(samples[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, path_a / Path(f"{cfg.experiment.name}_epoch_{epoch}_sample_{i:03d}.png"), normalize=False)


    print(f"--- Finished Training for {model_name} ---")
    ckpt_mgr.save(model, model_name)


class SuperDiffSampler:
    """Implements the SUPERDIFF algorithm for composing two pre-trained models."""

    def __init__(self, sde: VPSDE):
        self.sde = sde

    @torch.no_grad()
    def sample(self, model1, model2, batch_size, shape, device, operation='OR', temp=1.0, bias=0.0):
        model1.eval()
        model2.eval()
        x = torch.randn((batch_size, *shape), device=device)
        log_q1 = torch.zeros(batch_size, device=device)
        log_q2 = torch.zeros(batch_size, device=device)
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc=f"SUPERDIFF Sampling ({operation})", leave=False):
            t_idx = timesteps[i]
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            noise1, noise2 = model1(x, t.float()), model2(x, t.float())
            score1, score2 = -noise1 / sqrt_one_minus_alpha_bar_t, -noise2 / sqrt_one_minus_alpha_bar_t
            if operation.upper() == 'OR':
                logits = torch.stack([log_q1, log_q2], dim=1)
                kappas = F.softmax(temp * logits + bias, dim=1)
                kappa1, kappa2 = kappas[:, 0].view(-1, 1, 1, 1), kappas[:, 1].view(-1, 1, 1, 1)
            elif operation.upper() == 'AND':
                # Heuristic to balance log-densities, pushing towards an equal density state.
                # A rigorous implementation solves the linear system in Prop. 6 of the paper [cite: 207-210].
                probs = F.softmax(torch.stack([-log_q1, -log_q2], dim=1), dim=1)
                kappa1, kappa2 = probs[:, 0].view(-1, 1, 1, 1), probs[:, 1].view(-1, 1, 1, 1)
            else:
                kappa1, kappa2 = 0.5, 0.5
            combined_score = kappa1 * score1 + kappa2 * score2
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            mean = (1 / sqrt_alpha_t) * (x + beta_t * combined_score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1)
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            dx = x_prev - x
            dtau = 1.0 / self.sde.num_timesteps
            d = x.shape[1] * x.shape[2] * x.shape[3]
            div_f = -0.5 * beta_t.squeeze() * d

            def update_log_q(log_q, score):
                term1 = torch.sum(dx * score, dim=[1, 2, 3])
                f_term = -0.5 * beta_t * x
                g_sq_term = beta_t
                inner_prod_term = torch.sum((f_term - 0.5 * g_sq_term * score) * score, dim=[1, 2, 3])
                return log_q + term1 + (div_f + inner_prod_term) * dtau

            log_q1, log_q2 = update_log_q(log_q1, score1), update_log_q(log_q2, score2)
            x = x_prev
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_single_model(self, model, batch_size, shape, device):
        model.eval()
        x = torch.randn((batch_size, *shape), device=device)
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc="Single Model Sampling", leave=False):
            t_idx = timesteps[i]
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            noise = model(x, t.float())
            score = -noise / sqrt_one_minus_alpha_bar_t
            mean = (1 / sqrt_alpha_t) * (x + beta_t * score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1)
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            x = x_prev
        return x.clamp(-1, 1)


# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================
# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================
def visualize_results(samples_a, samples_b, superdiff_samples, ckpt_mgr):
    """Saves a grid comparing the three sets of samples to the correct directory."""
    def norm(s): return (s.clamp(-1, 1) + 1) / 2

    # Save individual model results
    path_a = ckpt_mgr.get_path( 'results') / model_A_name
    path_a.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(norm(samples_a), path_a / "samples_final.png", nrow=samples_a.shape[0])
    print(f"Saved Model A samples to {path_a}")

    path_b = ckpt_mgr.get_path('results') / model_B_name
    path_b.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(norm(samples_b), path_b / "samples_final.png", nrow=samples_b.shape[0])
    print(f"Saved Model B samples to {path_b}")

    # Save composition results
    path_comp = ckpt_mgr.get_path( 'results') / "composition"
    path_comp.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(norm(superdiff_samples), path_comp / "composition_final.png", nrow=superdiff_samples.shape[0])
    print(f"Saved Composed samples to {path_comp}")

    # Save comparison grid
    comparison_path = ckpt_mgr.get_path('results') / "composition"
    comparison_path.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(torch.cat([
        norm(samples_a), norm(samples_b), norm(superdiff_samples)
    ]), comparison_path / "comparison_grid.png" , nrow=samples_a.shape[0])
    print(f"Saved final comparison grid to {comparison_path} (Top: Model A, Middle: Model B, Bottom: Composed)")

def is_cluster():
    import socket, os
    hostname = socket.gethostname()
    return "mscluster" in hostname or "wits" in hostname or os.environ.get("IS_CLUSTER") == "1"

def tiny_subset(dataset: Dataset, num_items: int = 8) -> Subset:
    """Return a tiny subset of the dataset for quick overfitting checks."""
    indices = list(range(min(len(dataset), num_items)))
    return Subset(dataset, indices)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Create structured directories ---
    base_path = f"/gluster/mmolefe/PhD/{args.project_name}" if is_cluster() else "./"
    base_dir = Path(base_path)
    ckpt_mgr = CheckpointManager(base_dir, args.exp_name, args.run_id)

    # Create a log directory
    log_dir = base_dir / args.exp_name / args.run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved in: {log_dir}")

    # Create a log directory
    results_dir = base_dir / args.exp_name / args.run_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    sde = VPSDE(device=device)

    cfg = Box({
        "experiment": {
            "name": args.exp_name,
            "run": args.run_id,
        },
        "dataset": {
            "image_size": 32,
            "channels": 3,
            "dataset_a": args.dataset_a,
            "dataset_b": args.dataset_b,
            "content_digit": [7],
            "style_color": (1.0, 0.0, 0.0),  # Red
            "num_samples_color": 5000  # Define a fixed number for the color dataset
        },
        "training": {
            "do_train": not args.skip_train,
            "epochs": 1 if args.sanity else 100,
            "log_every_epoch": 1 if args.sanity else 20,
            "sanity_num_examples": 8,
            "batch_size": 4 if args.sanity else 128
        },
        "optimizer": {
            "params": {"lr": 2e-4}
        },
        "sampling": {
            "batch_size": 8
        }
    })

    print(f"--- 🧪 RUNNING EXPERIMENT: {args.exp_name} | RUN ID: {args.run_id} ---")

    # --- Corrected Dataset Initialization ---

    # Create Dataset A
    print(f"Initializing dataset A: {cfg.dataset.dataset_a}")
    if cfg.dataset.dataset_a == 'GrayscaleMNIST':
        dataset_A = GrayscaleMNIST(
            image_size=cfg.dataset.image_size,
            target_digits=cfg.dataset.content_digit
        )
        # Match the number of color samples to the number of digit samples
        num_style_samples = len(dataset_A)
    elif cfg.dataset.dataset_a == 'SimpleShapesDataset':
        num_style_samples = cfg.dataset.num_samples_color
        dataset_A = SimpleShapesDataset(
            image_size=cfg.dataset.image_size,
            num_samples=num_style_samples,
            shape_color=cfg.dataset.style_color
        )
    else:
        raise ValueError(f"Unknown dataset for dataset_a: {cfg.dataset.dataset_a}")

    # Create Dataset B
    print(f"Initializing dataset B: {cfg.dataset.dataset_b}")
    if cfg.dataset.dataset_b == 'SimpleShapesDataset':
        dataset_B = SimpleShapesDataset(
            image_size=cfg.dataset.image_size,
            num_samples=num_style_samples,
            shape_color=cfg.dataset.style_color
        )
    elif cfg.dataset.dataset_b == 'GrayscaleMNIST':
        dataset_B = GrayscaleMNIST(
            image_size=cfg.dataset.image_size,
            target_digits=cfg.dataset.content_digit
        )
    else:
        raise ValueError(f"Unknown dataset for dataset_b: {cfg.dataset.dataset_b}")

    if args.sanity:
        dataset_A = tiny_subset(dataset_A, cfg.training.sanity_num_examples)
        dataset_B = tiny_subset(dataset_B, cfg.training.sanity_num_examples)

    loader_A = DataLoader(dataset_A, batch_size=cfg.training.batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=cfg.training.batch_size, shuffle=True)
    operation = 'AND'

    model_A = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_B = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)

    if cfg.training.do_train:
        print("\n--- 🏋️ Starting Training Phase ---")
        train(cfg, model_A, sde, loader_A, device, model_A_name, ckpt_mgr, results_dir)
        train(cfg, model_B, sde, loader_B, device, model_B_name, ckpt_mgr, results_dir)
    else:
        print("\n--- Skipping Training Phase ---")

    print("\n--- 🎬 Starting Inference Phase ---")
    model_A = ckpt_mgr.load(model_A, model_A_name, device)
    model_B = ckpt_mgr.load(model_B, model_B_name, device)

    sampler = SuperDiffSampler(sde)
    shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)

    print("Generating samples from Model A (Content)...")
    samples_A = sampler.sample_single_model(model_A, cfg.sampling.batch_size, shape, device)

    print("Generating samples from Model B (Style)...")
    samples_B = sampler.sample_single_model(model_B, cfg.sampling.batch_size, shape, device)

    print(f"Generating composed samples using SUPERDIFF '{operation}'...")
    composed_samples = sampler.sample(model_A, model_B, cfg.sampling.batch_size, shape, device, operation)

    visualize_results(samples_A[:4], samples_B[:4], composed_samples[:4], ckpt_mgr)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose two diffusion models.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--run_id", type=str, required=True, help="A unique ID for the current run.")
    parser.add_argument("--project_name", type=str, default="mini-composable-diffusion-model",
                        help="Name of the project directory.")
    parser.add_argument("--dataset_a", type=str, default="GrayscaleMNIST",
                        choices=["GrayscaleMNIST", "SimpleShapesDataset"],
                        help="Dataset for the first model (content).")
    parser.add_argument("--dataset_b", type=str, default="SimpleShapesDataset",
                        choices=["GrayscaleMNIST", "SimpleShapesDataset"], help="Dataset for the second model (style).")
    parser.add_argument("--skip_train", action='store_true',
                        help="Skip training and load models directly from checkpoints.")
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)