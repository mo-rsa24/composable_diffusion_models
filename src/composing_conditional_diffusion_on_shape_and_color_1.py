import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import math
import os


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EXP_NAME =  "shape_and_color"
    IMG_SIZE = 64
    BATCH_SIZE = 128
    PREFIX = "samples"
    TIMESTEPS = 500
    NUM_EPOCHS = 200  # Increase for better results
    LR = 1e-4
    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]
    # Hold out a combination to test for true compositionality
    HOLDOUT_COMBINATION = ("triangle", "blue")
    OUTPUT_DIR = f"src/scripts/mini-experiments/visualizations/{EXP_NAME}/composable_diffusion_output_part_2"


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- 1. Dataset Generation ---

class ShapesDataset(Dataset):
    """
    A dataset that generates images of simple shapes with specified colors.
    The images are generated on-the-fly.
    """

    def __init__(self, size=1000, img_size=64, shapes=None, colors=None, holdout=None, train=True):
        self.size = size
        self.img_size = img_size
        self.shapes = shapes if shapes is not None else Config.SHAPES
        self.colors = colors if colors is not None else Config.COLORS
        self.holdout = holdout
        self.train = train

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]
        if self.holdout and self.train:
            self.all_combinations.remove(self.holdout)

        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

        self.transforms = Compose([
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),  # Scales to [0, 1]
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

    def __len__(self):
        return self.size

    def _draw_shape(self, shape, color_name, draw):
        """Helper function to draw a shape on a PIL image."""
        img_size = self.img_size
        margin = img_size // 4

        # Define coordinates
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
        # Choose a random combination
        shape_name, color_name = self.all_combinations[idx % len(self.all_combinations)]

        # Create a blank image
        image = Image.new("RGB", (self.img_size, self.img_size), "white")
        draw = ImageDraw.Draw(image)

        # Draw the shape
        self._draw_shape(shape_name, color_name, draw)

        shape_idx = self.shape_to_idx[shape_name]
        color_idx = self.color_to_idx[color_name]

        return self.transforms(image), torch.tensor(shape_idx), torch.tensor(color_idx)


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


# In the Block class
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
        # --- MODIFICATION: Replace BatchNorm2d with GroupNorm ---
        self.gn1 = nn.GroupNorm(8, out_ch)  # 8 groups is a common choice
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):  # Now we expect the embedding directly
        # First Conv
        h = self.gn1(self.relu(self.conv1(x)))

        # --- MODIFICATION: Project and add time/label embedding ---
        time_emb = self.relu(self.time_mlp(t_emb))
        time_emb = time_emb[(...,) + (None,) * 2]  # Reshape for broadcasting
        h = h + time_emb

        # Second Conv
        h = self.gn2(self.relu(self.conv2(h)))

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

    # In the SimpleUnet forward pass
    def forward(self, x, timestep, y):
        # Embedd time
        t_emb = self.time_mlp(timestep)
        # Embedd label
        y_emb = self.label_emb(y)
        # Combine embeddings
        combined_emb = t_emb + y_emb

        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            # --- MODIFICATION: Pass the combined embedding to each block ---
            x = down(x, combined_emb)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            # --- MODIFICATION: Pass the combined embedding to each block ---
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
            if debug:
                if step % dataloader.dataset.__len__():
                    from src.utils.visualization import visualize_images
                    visualize_images(images, denormalize=False)
            # Select the correct label for conditioning
            if condition_type == 'shape':
                labels = shape_labels.to(Config.DEVICE)
            else:  # 'color'
                labels = color_labels.to(Config.DEVICE)

            # Sample timesteps
            t = torch.randint(0, Config.TIMESTEPS, (batch_size,), device=Config.DEVICE).long()

            loss = p_losses(model, images, t, labels, loss_type="l1")
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        if epoch % 20 == 0:
            model.eval()
            generated_image = sample_image(model, labels)
            samples_dir: Path = Path(Config.OUTPUT_DIR) / condition_type / f"epoch_{epoch}"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for(i, img) in enumerate(generated_image[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, samples_dir / Path(f"{Config.PREFIX}_{i:03d}.png"), normalize=False)
    print(f"--- Finished training {condition_type.upper()} model ---")


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
    print(
        f"Sampling with composition: {Config.SHAPES[shape_idx]} ({w_shape:.1f}) + {Config.COLORS[color_idx]} ({w_color:.1f})")

    # Start from pure noise
    img_size = Config.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=Config.DEVICE)

    shape_label = torch.full((1,), shape_idx, device=Config.DEVICE, dtype=torch.long)
    color_label = torch.full((1,), color_idx, device=Config.DEVICE, dtype=torch.long)

    for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc="Composed Sampling", total=Config.TIMESTEPS):
        t = torch.full((1,), i, device=Config.DEVICE, dtype=torch.long)

        # Predict noise from each model
        pred_noise_shape = shape_model(img, t, shape_label)
        pred_noise_color = color_model(img, t, color_label)

        # Combine the noise predictions (this is the core composition step)
        composed_noise = (w_shape * pred_noise_shape + w_color * pred_noise_color) / (w_shape + w_color)

        # Use the composed noise to denoise for one step
        betas_t = extract(betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, img.shape)

        model_mean = sqrt_recip_alphas_t * (
                img - betas_t * composed_noise / sqrt_one_minus_alphas_cumprod_t
        )

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
    dataset = ShapesDataset(size=5000, holdout=Config.HOLDOUT_COMBINATION, train=True)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. Initialize Models
    shape_model = SimpleUnet(num_classes=len(Config.SHAPES)).to(Config.DEVICE)
    color_model = SimpleUnet(num_classes=len(Config.COLORS)).to(Config.DEVICE)

    # 3. Train Models
    shape_optimizer = torch.optim.Adam(shape_model.parameters(), lr=Config.LR)
    color_optimizer = torch.optim.Adam(color_model.parameters(), lr=Config.LR)

    train_model(shape_model, dataloader, shape_optimizer, Config.NUM_EPOCHS, 'shape')
    train_model(color_model, dataloader, color_optimizer, Config.NUM_EPOCHS, 'color')

    # --- 4. Perform Compositional Sampling ---
    print("\n--- Starting Compositional Sampling ---")

    # Get indices for all shapes and colors
    shape_map = {name: i for i, name in enumerate(Config.SHAPES)}
    color_map = {name: i for i, name in enumerate(Config.COLORS)}

    # Generate an image for every possible combination
    generated_images = []
    for s_name in Config.SHAPES:
        for c_name in Config.COLORS:
            s_idx = shape_map[s_name]
            c_idx = color_map[c_name]

            # Note if this combination was held out
            if (s_name, c_name) == Config.HOLDOUT_COMBINATION:
                print(f"\nGenerating HELD-OUT combination: {s_name}, {c_name}")
            else:
                print(f"\nGenerating seen combination: {s_name}, {c_name}")

            # Sample using composition
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

        plt.figure(figsize=(8, 8))
        plt.imshow(grid_pil)
        plt.axis('off')

        # Add labels
        plt.title("Compositional Generation Results\n(Rows: Shape, Cols: Color)")
        for i, shape in enumerate(Config.SHAPES):
            plt.text(-20, (i + 0.5) * Config.IMG_SIZE, shape, ha='center', va='center', rotation=90, fontsize=12)
        for i, color in enumerate(Config.COLORS):
            plt.text((i + 0.5) * Config.IMG_SIZE, -20, color, ha='center', va='center', fontsize=12)

        plt.show()
    except ImportError:
        print("Matplotlib not found. Cannot display image. Please view the saved file.")

