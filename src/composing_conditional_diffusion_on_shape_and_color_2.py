import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import math
import os


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- CHOOSE THE EXPERIMENT ---
    # Options: "shapes" or "mnist"
    DATASET_TYPE = "mnist"

    # --- General Settings ---
    IMG_SIZE = 64
    BATCH_SIZE = 128
    TIMESTEPS = 200
    NUM_EPOCHS = 200  # For good results, MNIST might need more epochs
    LR = 1e-4
    PREFIX = "samples"

    # --- MNIST Specific ---
    if DATASET_TYPE == "mnist":
        EXP_NAME = "mnist_and_color"
        # The "shapes" are now digits
        SHAPES = [str(i) for i in range(10)]
        COLORS = ["red", "green", "blue", "yellow"]  # Added more colors
        # Hold out the number '7' and the color 'blue'
        HOLDOUT_SHAPE = 7
        HOLDOUT_COLOR = "blue"
        HOLDOUT_COMBINATION = (str(HOLDOUT_SHAPE), HOLDOUT_COLOR)

    # --- Shape Specific ---
    elif DATASET_TYPE == "shapes":
        EXP_NAME = "shape_and_color"
        SHAPES = ["circle", "square", "triangle"]
        COLORS = ["red", "green", "blue"]
        # Hold out a shape/color combination for testing
        HOLDOUT_SHAPE = "triangle"
        HOLDOUT_COLOR = "blue"
        HOLDOUT_COMBINATION = (HOLDOUT_SHAPE, HOLDOUT_COLOR)

    OUTPUT_DIR = f"visualizations/{EXP_NAME}/digit_color_composition"


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- 1. Dataset Generation ---

class CompositionalDataset(Dataset):
    """
    A dataset that generates images for compositionality experiments.
    It can either generate simple colored shapes or create colored MNIST digits.
    """

    def __init__(self, size=10000, img_size=64, train=True):
        self.size = size
        self.img_size = img_size
        self.train = train
        self.dataset_type = Config.DATASET_TYPE

        self.shapes = Config.SHAPES
        self.colors = Config.COLORS
        self.holdout_shape = Config.HOLDOUT_SHAPE
        self.holdout_color = Config.HOLDOUT_COLOR

        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

        # Define transforms
        self.transforms = Compose([
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

        if self.dataset_type == "mnist":
            # Load MNIST dataset
            mnist_transforms = Compose([
                # MNIST is grayscale, need to convert to RGB for coloring
                Lambda(lambda x: x.convert("RGB"))
            ])
            self.mnist_data = MNIST(root='./data', train=self.train, download=True, transform=mnist_transforms)

            # Filter out the holdout digit if in training mode
            if self.train:
                indices = [i for i, (img, label) in enumerate(self.mnist_data) if label != self.holdout_shape]
                self.mnist_data = torch.utils.data.Subset(self.mnist_data, indices)

        # Filter out holdout color for the training set
        self.train_colors = self.colors
        if self.train:
            self.train_colors = [c for c in self.colors if c != self.holdout_color]

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]

    def __len__(self):
        if self.dataset_type == 'mnist':
            return len(self.mnist_data)
        return self.size

    def _draw_shape(self, shape, color_name, draw):
        """Helper function to draw a shape on a PIL image."""
        img_size = self.img_size
        margin = img_size // 4
        top_left = (margin, margin)
        bottom_right = (img_size - margin, img_size - margin)

        if shape == "circle":
            draw.ellipse([top_left, bottom_right], fill=color_name)
        elif shape == "square":
            draw.rectangle([top_left, bottom_right], fill=color_name)
        elif shape == "triangle":
            p1 = (img_size // 2, margin)
            p2 = (margin, img_size - margin)
            p3 = (img_size - margin, img_size - margin)
            draw.polygon([p1, p2, p3], fill=color_name)

    def __getitem__(self, idx):
        if self.dataset_type == "shapes":
            # Choose a random combination (excluding holdout for training)
            train_combinations = [c for c in self.all_combinations if c != Config.HOLDOUT_COMBINATION]
            combinations = train_combinations if self.train else self.all_combinations
            shape_name, color_name = combinations[idx % len(combinations)]

            image = Image.new("RGB", (self.img_size, self.img_size), "white")
            draw = ImageDraw.Draw(image)
            self._draw_shape(shape_name, color_name, draw)

            shape_idx = self.shape_to_idx[shape_name]
            color_idx = self.color_to_idx[color_name]

            return self.transforms(image), torch.tensor(shape_idx), torch.tensor(color_idx)

        elif self.dataset_type == "mnist":
            # Get MNIST image and its label (shape_label)
            mnist_img_pil, shape_label = self.mnist_data[idx]

            # Randomly select a color
            color_name = self.train_colors[idx % len(self.train_colors)]
            color_label = self.color_to_idx[color_name]

            # Create a colored digit on a white background
            background = Image.new("RGB", (self.img_size, self.img_size), "white")
            # The MNIST digit (black part) will be used as a mask to paste the color
            # We need to resize the digit to match our target image size
            mnist_img_pil = mnist_img_pil.resize((self.img_size, self.img_size))
            mask = mnist_img_pil.convert("L").point(lambda x: 255 if x < 128 else 0)  # Invert and binarize

            colored_shape = Image.new("RGB", (self.img_size, self.img_size), color_name)
            background.paste(colored_shape, (0, 0), mask)

            return self.transforms(background), torch.tensor(shape_label), torch.tensor(color_label)


# --- 2. Diffusion Logic (DDPM) ---

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


# Pre-calculate constants for the diffusion process
betas = linear_beta_schedule(timesteps=Config.TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    """Extracts the correct 'a' coefficient for a batch of timesteps 't'."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, noise=None):
    """Forward diffusion process: adds noise to an image."""
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, y, loss_type="l1"):
    """Calculates the loss for the denoising model."""
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
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
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.gn1(self.relu(self.conv1(x)))
        time_emb_proj = self.relu(self.time_mlp(t_emb))
        time_emb_proj = time_emb_proj[(...,) + (None,) * 2]
        h = h + time_emb_proj
        h = self.gn2(self.relu(self.conv2(h)))
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

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, y):
        t_emb = self.time_mlp(timestep)
        y_emb = self.label_emb(y)
        combined_emb = t_emb + y_emb

        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, combined_emb)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, combined_emb)
        return self.output(x)


# --- 4. Training ---

def train_model(model, dataloader, optimizer, num_epochs, condition_type, debug: bool = False):
    """Trains one of the specialist models."""
    print(f"--- Training {condition_type.upper()} model ---")
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, (images, shape_labels, color_labels) in enumerate(progress_bar):
            optimizer.zero_grad()

            batch_size = images.shape[0]
            images = images.to(Config.DEVICE)

            if condition_type == 'shape':
                labels = shape_labels.to(Config.DEVICE)
            else:  # 'color'
                labels = color_labels.to(Config.DEVICE)

            t = torch.randint(0, Config.TIMESTEPS, (batch_size,), device=Config.DEVICE).long()
            loss = p_losses(model, images, t, labels, loss_type="l1")
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        if epoch % 50 == 0:
            model.eval()
            generated_image = sample_image(model, labels)
            samples_dir: Path = Path(Config.OUTPUT_DIR) / condition_type / f"epoch_{epoch}"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for(i, img) in enumerate(generated_image[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, samples_dir / Path(f"{Config.PREFIX}_{i:03d}.png"), normalize=False)

# --- 5. Sampling / Inference ---
@torch.no_grad()
def sample_image(model: SimpleUnet, labels):
    """Sample using the composed scores of the two models."""
    print(f"Sampling Diffusion")

    # Start from pure noise
    img_size = Config.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=Config.DEVICE)

    for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc="Diffusion Sampling", total=Config.TIMESTEPS):
        t = torch.full((1,), i, device=Config.DEVICE, dtype=torch.long)
        predicted_noise = model(img, t, labels)

        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]

        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_img * (img - coeff_pred_noise * predicted_noise)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, img.shape)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    return img

@torch.no_grad()
def sample_composed(shape_model, color_model, shape_idx, color_idx, w_shape=1.0, w_color=1.0):
    """Sample using the composed scores of the two models."""
    shape_name = Config.SHAPES[shape_idx]
    color_name = Config.COLORS[color_idx]
    print(f"Sampling with composition: {shape_name} ({w_shape:.1f}) + {color_name} ({w_color:.1f})")

    img_size = Config.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=Config.DEVICE)

    shape_label = torch.full((1,), shape_idx, device=Config.DEVICE, dtype=torch.long)
    color_label = torch.full((1,), color_idx, device=Config.DEVICE, dtype=torch.long)

    for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc="Composed Sampling", total=Config.TIMESTEPS):
        t = torch.full((1,), i, device=Config.DEVICE, dtype=torch.long)

        pred_noise_shape = shape_model(img, t, shape_label)
        pred_noise_color = color_model(img, t, color_label)

        composed_noise = (w_shape * pred_noise_shape + w_color * pred_noise_color) / (w_shape + w_color)

        betas_t = extract(betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, img.shape)

        model_mean = sqrt_recip_alphas_t * (img - betas_t * composed_noise / sqrt_one_minus_alphas_cumprod_t)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, img.shape)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise
    return img


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Create Datasets and Dataloaders
    dataset = CompositionalDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. Initialize Models
    # The 'shape' model learns digits or geometric shapes
    shape_model = SimpleUnet(num_classes=len(Config.SHAPES)).to(Config.DEVICE)
    # The 'color' model learns colors
    color_model = SimpleUnet(num_classes=len(Config.COLORS)).to(Config.DEVICE)

    # 3. Train Models
    shape_optimizer = torch.optim.Adam(shape_model.parameters(), lr=Config.LR)
    color_optimizer = torch.optim.Adam(color_model.parameters(), lr=Config.LR)

    # Train the model to recognize shape/digit
    train_model(shape_model, dataloader, shape_optimizer, Config.NUM_EPOCHS, 'shape')
    # Train the model to recognize color
    train_model(color_model, dataloader, color_optimizer, Config.NUM_EPOCHS, 'color')

    # --- 4. Perform Compositional Sampling ---
    print("\n--- Starting Compositional Sampling ---")

    shape_map = {name: i for i, name in enumerate(Config.SHAPES)}
    color_map = {name: i for i, name in enumerate(Config.COLORS)}

    generated_images = []
    for s_name in Config.SHAPES:
        for c_name in Config.COLORS:
            s_idx = shape_map[s_name]
            c_idx = color_map[c_name]

            # Note if this combination was held out
            is_holdout = (s_name == str(
                Config.HOLDOUT_SHAPE) and c_name == Config.HOLDOUT_COLOR) if Config.DATASET_TYPE == "mnist" else (
                                                                                                                 s_name,
                                                                                                                 c_name) == Config.HOLDOUT_COMBINATION

            if is_holdout:
                print(f"\nGenerating HELD-OUT combination: {s_name}, {c_name}")
            else:
                print(f"\nGenerating seen combination: {s_name}, {c_name}")

            generated_image = sample_composed(shape_model, color_model, s_idx, c_idx, w_shape=1.0, w_color=1.0)
            generated_images.append(generated_image)

    # Save the results in a grid
    grid = make_grid(torch.cat(generated_images), nrow=len(Config.COLORS), normalize=True, value_range=(-1, 1))
    grid_pil = ToPILImage()(grid)

    output_path = os.path.join(Config.OUTPUT_DIR, "compositional_generation_grid.png")
    grid_pil.save(output_path)
    print(f"\nSaved generation grid to {output_path}")

    # Display the grid if in an interactive environment
    try:
        import matplotlib.pyplot as plt

        # Determine grid size based on number of shapes/colors
        fig_width = len(Config.COLORS) * 1.5
        fig_height = len(Config.SHAPES) * 1.5

        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(grid_pil)
        plt.axis('off')

        plt.title(f"Compositional Generation: {Config.DATASET_TYPE.title()}\n(Rows: Concept 1, Cols: Concept 2)")

        # Add row labels (shapes or digits)
        for i, shape in enumerate(Config.SHAPES):
            plt.text(-20, (i + 0.5) * Config.IMG_SIZE, shape, ha='center', va='center', rotation=90, fontsize=12)

        # Add column labels (colors)
        for i, color in enumerate(Config.COLORS):
            plt.text((i + 0.5) * Config.IMG_SIZE, -20, color, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not found. Cannot display image. Please view the saved file.")