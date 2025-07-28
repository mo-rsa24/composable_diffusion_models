import argparse
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
from tqdm import tqdm
import math
import os

from src.utils.tools import is_cluster, CheckpointManager, save_config_to_yaml, tiny_subset


# --- 1. Dataset Generation (Colored MNIST) ---

class ColoredMNIST(Dataset):
    """
    A dataset that loads MNIST digits and applies a specified color.
    """

    def __init__(self, cfg: Box, size=60000,holdout=None, train=True):
        self.train = train
        # Load the full MNIST dataset
        mnist_dataset = MNIST(root="./data", train=self.train, download=True, transform=ToTensor())

        # Limit the dataset size if specified
        self.data = mnist_dataset.data[:size]
        self.targets = mnist_dataset.targets[:size]

        self.colors = cfg.experiment.colors
        self.digits = cfg.experiment.digits
        self.holdout = holdout

        self.color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        }
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

        # Create a list of valid (digit, color_idx) pairs for training
        self.valid_indices = []
        for i in range(len(self.data)):
            digit = self.targets[i].item()
            for color_name in self.colors:
                if self.train and self.holdout and (digit, color_name) == self.holdout:
                    continue  # Skip the holdout combination for training data
                self.valid_indices.append((i, color_name))

        self.transforms = Compose([
            ToPILImage(),
            Resize(cfg.dataset.image_size),
            CenterCrop(cfg.dataset.image_size),
            ToTensor(),  # Scales to [0, 1]
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx, color_name = self.valid_indices[idx]

        # Get the grayscale MNIST image
        digit_img = self.data[img_idx]  # This is a grayscale tensor
        digit_label = self.targets[img_idx].item()

        # Convert grayscale to RGB by coloring the digit
        # Start with a black background
        rgb_img = torch.zeros(3, digit_img.shape[0], digit_img.shape[1])
        # Get the color as a tensor [R, G, B] normalized to [0, 1]
        color_tensor = torch.tensor([c / 255.0 for c in self.color_map[color_name]])

        # Apply color where the digit is white
        for i in range(3):
            rgb_img[i][digit_img > 0] = color_tensor[i]

        color_label = self.color_to_idx[color_name]

        return self.transforms(rgb_img), torch.tensor(digit_label), torch.tensor(color_label)


# --- 2. Diffusion Logic (DDPM) ---

def linear_beta_schedule(timesteps, device = 'cpu'):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


# Pre-calculate constants for the diffusion process
class Sampler:
    def __init__(self, cfg: Box, device='cpu'):
        self.num_timesteps = cfg.diffusion.timesteps
        self.betas = linear_beta_schedule(timesteps=self.num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @torch.no_grad()
    def sample_single_model(self, model, condition_label_idx, batch_size, shape, device):
        model.eval()
        x = torch.randn((batch_size, *shape), device=device)
        timesteps = torch.arange(self.num_timesteps - 1, -1, -1, device=device)
        condition_label = torch.full((1,), condition_label_idx, device=device, dtype=torch.long)
        for i in tqdm(range(self.num_timesteps), desc="Single Model Sampling", leave=False):
            t_idx = timesteps[i]
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            noise = model(x, t.float(), condition_label)
            score = -noise / sqrt_one_minus_alpha_bar_t
            mean = (1 / sqrt_alpha_t) * (x + beta_t * score)
            if i < self.num_timesteps - 1:
                posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            x = x_prev
        return x.clamp(-1, 1)

def extract(a, t, x_shape):
    """Extracts the correct 'a' coefficient for a batch of timesteps 't'."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, sampler: Sampler, noise=None):
    """Forward diffusion process: adds noise to an image."""
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sampler.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sampler.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, y, sampler: Sampler, loss_type="l1"):
    """Calculates the loss for the denoising model."""
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, sampler=sampler, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, y)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# --- 3. U-Net Model ---

class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes timestep information."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """A basic convolutional block with GroupNorm."""

    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bn1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bn2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SimpleUnet(nn.Module):
    """A simplified U-Net for denoising."""

    def __init__(self, num_classes=10):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Class embedding
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                          time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], \
                                        time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, y):
        # Embedd time
        t = self.time_mlp(timestep)
        # Embedd label
        y_emb = self.label_emb(y)
        # Combine embeddings
        t = t + y_emb

        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


# --- 4. Training ---

def train_model(cfg, model, sampler:Sampler, dataloader, optimizer, num_epochs, condition_type, model_name, device, ckpt_mgr, results_dir):
    """Trains one of the specialist models."""
    print(f"--- Training {condition_type.upper()} model ---")
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, (images, digit_labels, color_labels) in enumerate(progress_bar):
            optimizer.zero_grad()

            batch_size = images.shape[0]
            images = images.to(device)

            # Select the correct label for conditioning
            if condition_type == 'digit':
                labels = digit_labels.to(device)
            else:  # 'color'
                labels = color_labels.to(device)

            # Sample timesteps
            t = torch.randint(0, cfg.diffusion.timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, images, t, labels, sampler, loss_type="l1")
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
        if epoch % cfg.training.log_every_epoch == 0:
            model.eval()
            shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)
            label_idx = labels[random.randint(0, len(labels)-1)]
            generated_image = sampler.sample_single_model(model, label_idx, cfg.sampling.batch_size, shape, device)

            path_a: Path = results_dir / model_name / f"epoch_{epoch}"
            path_a.mkdir(parents=True, exist_ok=True)
            for (i, img) in enumerate(generated_image[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, path_a / Path(f"{cfg.experiment.name}_epoch_{epoch}_sample_{i:03d}.png"),
                           normalize=False)
    print(f"--- Finished training {condition_type.upper()} model ---")
    ckpt_mgr.save(model, model_name)

# --- 5. Sampling / Inference ---

@torch.no_grad()
def sample_composed(cfg, digit_model, color_model, sampler: Sampler,  digit_idx, color_idx, device, w_digit=1.0, w_color=1.0):
    """Sample using the composed scores of the two models."""
    color_name = cfg.experiment.colors[color_idx]
    print(f"Sampling with composition: Digit {digit_idx} ({w_digit:.1f}) + Color {color_name} ({w_color:.1f})")

    # Start from pure noise
    img_size = cfg.dataset.image_size
    img = torch.randn((1, 3, img_size, img_size), device=device)

    digit_label = torch.full((1,), digit_idx, device=device, dtype=torch.long)
    color_label = torch.full((1,), color_idx, device=device, dtype=torch.long)

    for i in tqdm(reversed(range(0, cfg.diffusion.timesteps)), desc="Composed Sampling", total=cfg.diffusion.timesteps):
        t = torch.full((1,), i, device=device, dtype=torch.long)

        # Predict noise from each model
        pred_noise_digit = digit_model(img, t, digit_label)
        pred_noise_color = color_model(img, t, color_label)

        # Combine the noise predictions (this is the core composition step)
        composed_noise = (w_digit * pred_noise_digit + w_color * pred_noise_color) / (w_digit + w_color)

        # Use the composed noise to denoise for one step
        betas_t = extract(sampler.betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sampler.sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(sampler.sqrt_recip_alphas, t, img.shape)

        model_mean = sqrt_recip_alphas_t * (
                img - betas_t * composed_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = extract(sampler.posterior_variance, t, img.shape)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    return img


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


    cfg = Box({
        "experiment": {
            "name": args.exp_name,
            "run": args.run_id,
            "digits": list(range(10)),
            "colors": ["red", "green", "blue"]
        },
        "diffusion": {
            "timesteps": 1000
        },
        "dataset": {"image_size": 64, "channels": 3, "split_A_classes": list(range(5)),
                    "split_B_classes": list(range(5, 10))},
        "training": {"do_train": True,
                     "epochs": 1 if args.sanity else 100,
                     "log_every_epoch": 1 if args.sanity else 20,
                     "sanity_num_examples": 8,
                     "batch_size": 4 if args.sanity else 128
                     },
        "optimizer": {"params": {"lr": args.lr}},
        "sampling": {"batch_size": 8, "temp": 1.0}
    })
    save_config_to_yaml(cfg, log_dir)


    DIGITS = list(range(10))
    COLORS = ["red", "green", "blue"]
    # Hold out a combination to test for true compositionality
    HOLDOUT_COMBINATION = (7, "blue")  # (digit, color_name)

    dataset = ColoredMNIST(cfg, size=20000, holdout=HOLDOUT_COMBINATION,train=True)  # Use a subset for faster training
    if args.sanity:
        dataset = tiny_subset(dataset, cfg.training.sanity_num_examples)

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)

    # 2. Initialize Models
    digit_model = SimpleUnet(num_classes=len(DIGITS)).to(device)
    color_model = SimpleUnet(num_classes=len(COLORS)).to(device)

    # 3. Train Models
    digit_optimizer = torch.optim.Adam(digit_model.parameters(), lr=cfg.optimizer.params.get('lr'))
    color_optimizer = torch.optim.Adam(color_model.parameters(), lr=cfg.optimizer.params.get('lr'))
    sampler = Sampler(cfg, device=device)
    model_A_name = f"Digit"
    model_B_name = f"Color"
    train_model(cfg, digit_model, sampler,  dataloader, digit_optimizer, cfg.training.epochs, model_A_name, 'digit', device, ckpt_mgr, results_dir)
    train_model(cfg, color_model, sampler, dataloader, color_optimizer, cfg.training.epochs, model_B_name, 'color', device, ckpt_mgr, results_dir)

    # --- 4. Perform Compositional Sampling ---
    print("\n--- Starting Compositional Sampling ---")

    color_map = {name: i for i, name in enumerate(COLORS)}

    # Generate an image for every possible combination
    generated_images = []
    for d_idx in DIGITS:
        for c_name in COLORS:
            c_idx = color_map[c_name]

            # Note if this combination was held out
            if (d_idx, c_name) == HOLDOUT_COMBINATION:
                print(f"\nGenerating HELD-OUT combination: Digit {d_idx}, Color {c_name}")
            else:
                print(f"\nGenerating seen combination: Digit {d_idx}, Color {c_name}")

            # Sample using composition
            generated_image = sample_composed(cfg, digit_model, color_model, sampler, d_idx, c_idx, device, w_digit=1.0, w_color=1.0)
            generated_images.append(generated_image)

    # Save the results in a grid
    grid = make_grid(torch.cat(generated_images), nrow=len(COLORS), normalize=True, value_range=(-1, 1))
    grid_pil = ToPILImage()(grid)

    # output_path = os.path.join(Config.OUTPUT_DIR, "compositional_mnist_grid.png")
    path_comp = ckpt_mgr.get_path('results') / "composition"
    path_comp.mkdir(parents=True, exist_ok=True)
    grid_pil.save(path_comp / "composition_final.png")
    print(f"\nSaved generation grid to {path_comp / 'composition_final.png'}")

    # Display the grid if in an interactive environment
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 16))
        plt.imshow(grid_pil)
        plt.axis('off')

        # Add labels
        plt.title("Compositional Generation Results\n(Rows: Digit, Cols: Color)")
        for i, digit in enumerate(DIGITS):
            plt.text(-20, (i + 0.5) * cfg.dataset.image_size, str(digit), ha='center', va='center', rotation=90, fontsize=12)
        for i, color in enumerate(COLORS):
            plt.text((i + 0.5) * cfg.dataset.image_size, -20, color, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not found. Cannot display image. Please view the saved file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose two diffusion models.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--run_id", type=str, required=True, help="A unique ID for the current run.")
    parser.add_argument("--project_name", type=str, default="mini-composable-diffusion-model",
                        help="Name of the project directory.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)
