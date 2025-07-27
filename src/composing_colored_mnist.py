
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import torchvision
from box import Box
import math
from pathlib import Path
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm
import numpy as np


# ==============================================================================
# 1. MODEL AND SDE DEFINITIONS (Unchanged)
# ==============================================================================

class VPSDE:
    """Variance Preserving SDE noise schedule."""

    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, num_timesteps: int = 1000, device='cpu'):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_min, beta_max, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
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


class ColoredMNISTScoreModel(nn.Module):
    # This U-Net is flexible enough for both 32x32 CIFAR and 32x32 MNIST
    def __init__(self, in_channels: int = 3, time_emb_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
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
        # (Implementation is unchanged)
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
# 2. CUSTOM COLORED MNIST DATASET
# ==============================================================================
class ColoredMNIST(Dataset):
    """
    A wrapper for MNIST that colors each digit a specific color and filters by digit.
    """

    def __init__(self, image_size, target_digits=None):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mnist_dataset = datasets.MNIST(root='../../data', train=True, download=True)
        self.target_digits = target_digits
        if self.target_digits:
            self.indices = [i for i, (_, label) in enumerate(self.mnist_dataset) if label in self.target_digits]
        else:
            self.indices = list(range(len(self.mnist_dataset)))

        # Define a unique color for each digit (0-9)
        self.color_map = {
            0: (0.5, 0.5, 0.5),  # Gray
            1: (0.0, 0.5, 1.0),  # Light Blue
            2: (0.0, 0.8, 0.0),  # Green
            3: (0.0, 0.8, 0.8),  # Cyan
            4: (1.0, 0.5, 0.0),  # Orange
            5: (1.0, 1.0, 0.0),  # Yellow
            6: (1.0, 0.0, 0.0),  # Red
            7: (1.0, 0.0, 1.0),  # Magenta
            8: (0.5, 0.0, 1.0),  # Purple
            9: (0.6, 0.3, 0.1),  # Brown
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the original image and label
        original_idx = self.indices[idx]
        image, label = self.mnist_dataset[original_idx]

        # Apply transformations (resize, to tensor)
        image_tensor = self.transform(image)

        # Convert grayscale to 3-channel RGB by adding color
        color = self.color_map[label]
        color_tensor = torch.tensor(color).view(3, 1, 1)

        # The image tensor is white digit on black background (0 to 1).
        # We multiply it by the color to make the digit colored.
        colored_image = image_tensor.repeat(3, 1, 1) * color_tensor

        # Normalize to [-1, 1]
        final_image = (colored_image * 2) - 1

        return final_image, label


# ==============================================================================
# 3. TRAINING AND SAMPLING (Mostly Unchanged)
# ==============================================================================

class CheckpointManager:
    """A simple checkpoint manager."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, model_name):
        save_path = self.checkpoint_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")

    def load(self, model, model_name, device):
        load_path = self.checkpoint_dir / f"{model_name}.pth"
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint {load_path} not found.")
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded model checkpoint from {load_path}")
        return model


def train(cfg, model, vpsde_sampler, sde, train_loader, device, model_name, ckpt_mgr):
    """Simplified training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.params.lr)
    print(f"--- Starting Training for {model_name} ---")

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        for step, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}"), 1):
            optimizer.zero_grad();
            x0 = images.to(device)
            t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            sqrt_alpha_bar_t = sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
            predicted_noise = model(xt, t.float())
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward();
            optimizer.step()
        if epoch % cfg.training.epochs == 0:
            model.eval()
            shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)
            generated_image = vpsde_sampler.sample_single_model(model, cfg.sampling.batch_size, shape, device)
            OUTPUT_DIR = f"visualizations/{cfg.exp_name}/composing_colored_mnist"
            samples_dir: Path = Path(OUTPUT_DIR) / model_name / f"epoch_{epoch}"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for(i, img) in enumerate(generated_image[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, samples_dir / Path(f"{cfg.PREFIX}_{i:03d}.png"), normalize=False)
    print(f"--- Finished Training for {model_name} ---")
    ckpt_mgr.save(model, model_name)


class SuperDiffSampler:
    """Implements the SUPERDIFF algorithm for composing two pre-trained models."""

    def __init__(self, sde: VPSDE):
        self.sde = sde

    @torch.no_grad()
    def sample(self, model1, model2, batch_size, shape, device, operation='OR', temp=1.0, bias=0.0):
        # (Implementation is unchanged)
        model1.eval();
        model2.eval()
        x = torch.randn((batch_size, *shape), device=device)
        log_q1 = torch.zeros(batch_size, device=device);
        log_q2 = torch.zeros(batch_size, device=device)
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc=f"SUPERDIFF Sampling ({operation})", leave=False):
            t_idx = timesteps[i];
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            noise1, noise2 = model1(x, t.float()), model2(x, t.float());
            score1, score2 = -noise1 / sqrt_one_minus_alpha_bar_t, -noise2 / sqrt_one_minus_alpha_bar_t
            if operation.upper() == 'OR':
                logits = torch.stack([log_q1, log_q2], dim=1);
                kappas = F.softmax(temp * logits + bias, dim=1);
                kappa1, kappa2 = kappas[:, 0].view(-1, 1, 1, 1), kappas[:, 1].view(-1, 1, 1, 1)
            else:
                kappa1, kappa2 = 0.5, 0.5
            combined_score = kappa1 * score1 + kappa2 * score2
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1);
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            mean = (1 / sqrt_alpha_t) * (x + beta_t * combined_score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1);
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            dx = x_prev - x;
            dtau = 1.0 / self.sde.num_timesteps;
            d = x.shape[1] * x.shape[2] * x.shape[3];
            div_f = -0.5 * beta_t.squeeze() * d

            def update_log_q(log_q, score):
                term1 = torch.sum(dx * score, dim=[1, 2, 3]);
                f_term = -0.5 * beta_t * x;
                g_sq_term = beta_t
                inner_prod_term = torch.sum((f_term - 0.5 * g_sq_term * score) * score, dim=[1, 2, 3])
                return log_q + term1 + (div_f + inner_prod_term) * dtau

            log_q1, log_q2 = update_log_q(log_q1, score1), update_log_q(log_q2, score2);
            x = x_prev
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_single_model(self, model, batch_size, shape, device):
        # (Implementation is unchanged)
        model.eval();
        x = torch.randn((batch_size, *shape), device=device);
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc="Single Model Sampling", leave=False):
            t_idx = timesteps[i];
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1);
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1);
            noise = model(x, t.float())
            score = -noise / sqrt_one_minus_alpha_bar_t;
            mean = (1 / sqrt_alpha_t) * (x + beta_t * score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1);
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            x = x_prev
        return x.clamp(-1, 1)


# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================

### CHOOSE THE EXPERIMENT TO RUN ###
EXPERIMENT_TO_RUN = 'MNIST'  # Can be 'CIFAR10' or 'MNIST'


def get_cifar10_split_loader(classes, batch_size, image_size):
    """Creates a DataLoader for a specific subset of CIFAR-10 classes."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = [i for i, (_, label) in enumerate(full_dataset) if label in classes]
    return DataLoader(Subset(full_dataset, indices), batch_size=batch_size, shuffle=True)


def visualize_results(samples_a, samples_b, superdiff_samples, output_path):
    """Saves a grid comparing the three sets of samples."""

    def norm(samples): return (samples.clamp(-1, 1) + 1) / 2

    grid_a = torchvision.utils.make_grid(norm(samples_a), nrow=4)
    grid_b = torchvision.utils.make_grid(norm(samples_b), nrow=4)
    grid_superdiff = torchvision.utils.make_grid(norm(superdiff_samples), nrow=4)
    max_h = max(grid_a.shape[1], grid_b.shape[1], grid_superdiff.shape[1])

    def pad(grid): return F.pad(grid, (0, 0, 0, max_h - grid.shape[1])) if max_h > grid.shape[1] else grid

    combined_grid = torch.cat([pad(grid_a), pad(grid_b), pad(grid_superdiff)], dim=1)
    torchvision.utils.save_image(combined_grid, output_path)
    print(f"Saved final comparison to {output_path}")


if __name__ == '__main__':
    # --- Global Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dirs = {'ckpt': Path('./checkpoints'), 'viz': Path('./visualizations')}
    dirs['ckpt'].mkdir(exist_ok=True);
    dirs['viz'].mkdir(exist_ok=True)
    ckpt_mgr = CheckpointManager(checkpoint_dir=dirs['ckpt'])
    sde = VPSDE(device=device)
    vpsde_sampler = SuperDiffSampler(sde)
    # --- Experiment-Specific Configurations ---
    if EXPERIMENT_TO_RUN.upper() == 'CIFAR10':
        cfg = Box({
            "exp_name": "cifar10_split",
            "dataset": {"image_size": 32, "channels": 3, "split_A_classes": list(range(5)),
                        "split_B_classes": list(range(5, 10))},
            "training": {"do_train": True, "epochs": 10, "batch_size": 64}, "optimizer": {"params": {"lr": 2e-4}},
            "sampling": {"batch_size": 8, "temp": 1.0}
        })
        print("--- RUNNING CIFAR-10 EXPERIMENT ---")
        loader_A = get_cifar10_split_loader(cfg.dataset.split_A_classes, cfg.training.batch_size,
                                            cfg.dataset.image_size)
        loader_B = get_cifar10_split_loader(cfg.dataset.split_B_classes, cfg.training.batch_size,
                                            cfg.dataset.image_size)

    elif EXPERIMENT_TO_RUN.upper() == 'MNIST':
        cfg = Box({
            "exp_name": "mnist_colored",
            "dataset": {"image_size": 32, "channels": 3, "split_A_digit": [6], "split_B_digit": [2]},
            "PREFIX": "samples",
            # Red 6s and Green 2s
            "training": {"do_train": True, "epochs": 1, "batch_size": 128}, "optimizer": {"params": {"lr": 2e-4}},
            "sampling": {"batch_size": 8, "temp": 1.0}
        })
        print("--- RUNNING COLORED MNIST EXPERIMENT ---")
        dataset_A = ColoredMNIST(cfg.dataset.image_size, target_digits=cfg.dataset.split_A_digit)
        dataset_B = ColoredMNIST(cfg.dataset.image_size, target_digits=cfg.dataset.split_B_digit)
        loader_A = DataLoader(dataset_A, batch_size=cfg.training.batch_size, shuffle=True)
        loader_B = DataLoader(dataset_B, batch_size=cfg.training.batch_size, shuffle=True)

    else:
        raise ValueError("EXPERIMENT_TO_RUN must be 'CIFAR10' or 'MNIST'")

    # --- Model Initialization ---
    model_A = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_B = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_A_name = f"digit_6_model_A_{cfg.exp_name}"
    model_B_name = f"digit_2_model_B_{cfg.exp_name}"

    # --- Training Phase ---
    if cfg.training.do_train:
        train(cfg, model_A, vpsde_sampler, sde, loader_A, device, model_A_name, ckpt_mgr)
        train(cfg, model_B, vpsde_sampler, sde, loader_B, device, model_B_name, ckpt_mgr)

    # --- Inference and Composition Phase ---
    print("\n--- Starting Inference Phase ---")
    try:
        model_A = ckpt_mgr.load(model_A, model_A_name, device)
        model_B = ckpt_mgr.load(model_B, model_B_name, device)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please train the models first by setting `do_train=True`.")
        exit()

    shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)

    samples_A = vpsde_sampler.sample_single_model(model_A, cfg.sampling.batch_size, shape, device)
    samples_B = vpsde_sampler.sample_single_model(model_B, cfg.sampling.batch_size, shape, device)
    superdiff_samples = vpsde_sampler.sample(model_A, model_B, cfg.sampling.batch_size, shape, device, 'OR',
                                             temp=cfg.sampling.temp)

    # --- Visualization ---
    output_path = dirs['viz'] / f"experiment_{cfg.exp_name}_results.png"
    visualize_results(samples_A, samples_B, superdiff_samples, output_path)