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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.utils.tools import tiny_subset


# --- Configuration ---
class Config:
    """
    Configuration class for all hyperparameters and settings.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SANITY = True
    SANITY_NUM_EXAMPLES = 8
    EXP_NAME = "superdiff_composition"
    IMG_SIZE = 64
    BATCH_SIZE = 4 if SANITY else 128
    TIMESTEPS = 500
    NUM_EPOCHS =  1 if SANITY else 200
    LOG_EVERY_EPOCH = 1 if SANITY else 20
    LR = 2e-4
    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]
    # Hold out a combination to test for true compositionality
    HOLDOUT_COMBINATION = ("triangle", "blue")
    OUTPUT_DIR = f"./superdiff_composition_{EXP_NAME}_output"


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"Using device: {Config.DEVICE}")
print(f"Output will be saved to: {Config.OUTPUT_DIR}")


# --- 1. Dataset Generation ---

class ShapesDataset(Dataset):
    """
    A dataset that generates images of simple shapes with specified colors.
    It ensures that the holdout combination is excluded from the training set.
    """

    def __init__(self, config, length=10000, is_train=True):
        self.config = config
        self.length = length
        self.is_train = is_train
        self.img_size = config.IMG_SIZE
        self.shapes = config.SHAPES
        self.colors = config.COLORS
        self.holdout = config.HOLDOUT_COMBINATION

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]
        if self.is_train:
            self.allowed_combinations = [
                (s, c) for s, c in self.all_combinations if (s, c) != self.holdout
            ]
        else:
            self.allowed_combinations = self.all_combinations

        self.transforms = Compose([
            Resize((self.img_size, self.img_size)),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        combo_idx = torch.randint(0, len(self.allowed_combinations), (1,)).item()
        shape_name, color_name = self.allowed_combinations[combo_idx]

        shape_idx = self.shapes.index(shape_name)
        color_idx = self.colors.index(color_name)

        img = self.draw_shape(shape_name, color_name)
        return self.transforms(img), torch.tensor(shape_idx), torch.tensor(color_idx)

    def draw_shape(self, shape, color):
        img = Image.new('RGB', (self.img_size, self.img_size), color='black')
        draw = ImageDraw.Draw(img)

        s = self.img_size
        padding = s // 4

        if shape == 'circle':
            draw.ellipse([(padding, padding), (s - padding, s - padding)], fill=color)
        elif shape == 'square':
            draw.rectangle([(padding, padding), (s - padding, s - padding)], fill=color)
        elif shape == 'triangle':
            points = [(s / 2, padding), (s - padding, s - padding), (padding, s - padding)]
            draw.polygon(points, fill=color)

        return img


# --- 2. Diffusion Model Components ---

def get_linear_beta_schedule(timesteps):
    """Returns a linear beta schedule."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSDE:
    """
    Helper class for the diffusion process, framed as a Stochastic Differential Equation (SDE).
    This class contains methods for the forward (noising) and reverse (denoising) processes,
    and pre-calculates coefficients required for the Itô Density Estimator.
    """

    def __init__(self, timesteps, img_dims, device):
        self.timesteps = timesteps
        self.device = device
        self.img_dims = img_dims  # (C, H, W)

        # --- Standard DDPM schedule ---
        self.betas = get_linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # --- SDE coefficients for Itô Density Estimator (from SUPERDIFF paper) ---
        # The forward process is an Ornstein-Uhlenbeck SDE:
        # dx_t = f_t(x_t)dt + g_t dW_t
        # We pre-calculate the coefficients f_t and g_t^2.

        # Map DDPM schedule to SDE schedule (α_t in paper is sqrt_alphas_cumprod)
        log_alpha_t = 0.5 * torch.log(self.alphas_cumprod)
        log_sigma_t = 0.5 * torch.log(1. - self.alphas_cumprod)

        # Approximate derivatives using finite differences: dF/dt ≈ (F(t) - F(t-1)) / (1/T)
        # We pad to handle t=0 case
        f_t_coeff_ = (log_alpha_t - F.pad(log_alpha_t[:-1], (1, 0))) * self.timesteps
        self.f_t_coeff = f_t_coeff_.to(device)

        d_log_sigma_alpha_dt = ((log_sigma_t - log_alpha_t) - F.pad((log_sigma_t - log_alpha_t)[:-1],
                                                                    (1, 0))) * self.timesteps
        self.g_t_sq = (2 * (1. - self.alphas_cumprod) * d_log_sigma_alpha_dt).to(device)

        # Divergence of the drift f_t(x) = f_t_coeff * x is d * f_t_coeff
        self.div_f_t = (np.prod(self.img_dims) * self.f_t_coeff).to(device)

    def _extract(self, a, t, x_shape):
        """Extracts the appropriate t index for a batch of indices."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """Forward process: noise an image to a given timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model_output, x, t):
        """Reverse process: denoise an image by one step using DDPM formulation."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_losses(self, denoise_model, x_0, t, c, noise=None):
        """Calculates the training loss."""
        if noise is None:
            noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, c)
        return F.mse_loss(noise, predicted_noise)


# --- 3. UNet Model Architecture (Identical to previous version) ---

class Block(nn.Module):
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
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
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


class UNet(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = image_channels
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

    def forward(self, x, timestep, labels):
        t = self.time_mlp(timestep)
        c = self.label_emb(labels)
        t = t + c
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


# --- 4. Training Function (Identical to previous version) ---

def train_model(model, dataloader, diffusion,  optimizer, num_epochs, cond_type, config: Config):
    print(f"\n--- Training {cond_type.upper()} model ---")
    best_loss = float('inf')
    model_path = os.path.join(config.OUTPUT_DIR, f"best_{cond_type}_model.pth")
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, (images, shape_labels, color_labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            images = images.to(config.DEVICE)
            labels = (shape_labels if cond_type == 'shape' else color_labels).to(config.DEVICE)
            t = torch.randint(0, config.TIMESTEPS, (images.shape[0],), device=config.DEVICE).long()
            loss = diffusion.p_losses(model, images, t, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        if epoch % config.LOG_EVERY_EPOCH == 0:
            model.eval()
            generated_image = sample_image(model, diffusion, labels)
            samples_dir: Path = Path(Config.OUTPUT_DIR) / cond_type / f"epoch_{epoch}"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for(i, img) in enumerate(generated_image[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, samples_dir / Path(f"samples_{i:03d}.png"), normalize=False)
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")


# --- 5. SUPERDIFF Sampling and Visualization ---
@torch.no_grad()
def sample_image(model: UNet, diffusion: DiffusionSDE, labels):
    """Sample using the composed scores of the two models."""
    print(f"Sampling Diffusion")

    # Start from pure noise
    img_size = Config.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=Config.DEVICE)

    for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc="Diffusion Sampling", total=Config.TIMESTEPS):
        t = torch.full((1,), i, device=Config.DEVICE, dtype=torch.long)
        predicted_noise = model(img, t, labels)

        alpha_t = diffusion.alphas[i]
        alpha_cumprod_t = diffusion.alphas_cumprod[i]

        coeff_img = 1.0 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        model_mean = coeff_img * (img - coeff_pred_noise * predicted_noise)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = diffusion._extract(diffusion.posterior_variance, t, img.shape)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    return img

@torch.no_grad()
def sample_superdiff(shape_model, color_model, diffusion,
                     shape_class_idx, color_class_idx,
                     num_images=1, strategy='OR', temp=1.0, bias=0.0):
    """
    Generates images by composing two models using the SUPERDIFF algorithm.
    This function implements Algorithm 1 from the paper for the 'OR' operation.

    It works by:
    1. Adaptively reweighting the models' predicted scores (noise) at each step.
    2. The weights (kappa) are determined by the estimated log-densities of the current sample
       under each model.
    3. The log-densities are tracked throughout the reverse process using the novel
       Itô Density Estimator (Theorem 1), which avoids expensive computations.
    """
    device = diffusion.device
    x = torch.randn((num_images, 3, Config.IMG_SIZE, Config.IMG_SIZE), device=device)
    c_shape = torch.tensor([shape_class_idx] * num_images, device=device)
    c_color = torch.tensor([color_class_idx] * num_images, device=device)

    # Initialize log-densities for each model to zero
    log_q_shape = torch.zeros(num_images, 1, 1, 1, device=device)
    log_q_color = torch.zeros(num_images, 1, 1, 1, device=device)

    dt = 1.0 / diffusion.timesteps

    for i in tqdm(range(diffusion.timesteps - 1, -1, -1), desc=f"SUPERDIFF Sampling ({strategy})", leave=False):
        t = torch.full((num_images,), i, device=device, dtype=torch.long)
        x_t = x

        # --- 1. Adaptive Reweighting ---
        if strategy == 'OR':
            # For the OR operation, weights are a softmax of the current log-densities.
            # κ_τ^i ← softmax(T * log q_t^i(x_τ) + l)
            log_qs = torch.cat([log_q_shape, log_q_color], dim=1).squeeze()
            kappa = F.softmax(temp * log_qs + bias, dim=1)
            k_shape = kappa[:, 0].view(num_images, 1, 1, 1)
            k_color = kappa[:, 1].view(num_images, 1, 1, 1)
        else:
            # The 'AND' strategy requires solving a system of linear equations (Prop. 6)
            # which is more involved. We default to simple averaging for non-'OR' cases.
            k_shape = k_color = 0.5

        # --- 2. Vector Field Composition ---
        eps_shape = shape_model(x_t, t, c_shape)
        eps_color = color_model(x_t, t, c_color)

        # Composed noise: ε_composed = κ_shape * ε_shape + κ_color * ε_color
        eps_composed = k_shape * eps_shape + k_color * eps_color

        # --- 3. Denoising Step ---
        # Get the denoised sample x_{t-1} using the standard DDPM reverse step
        x_prev = diffusion.p_sample(eps_composed, x_t, t)

        # The change in x over this step
        dx = x_prev - x_t

        # --- 4. Itô Density Estimation (Update log-densities for the *next* step) ---
        # Implements the SDE from Theorem 1:
        # d log q_i = ⟨dx_τ, ∇ log q_i⟩ + (⟨∇, f_t⟩ + ⟨f_t - (g_t^2/2)∇ log q_i, ∇ log q_i⟩) dτ

        sigma_t = diffusion._extract(diffusion.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        score_shape = -eps_shape / (sigma_t + 1e-8)
        score_color = -eps_color / (sigma_t + 1e-8)

        f_t_coeff_t = diffusion._extract(diffusion.f_t_coeff, t, x_t.shape)
        f_t = f_t_coeff_t * x_t
        div_f_t_t = diffusion._extract(diffusion.div_f_t, t, x_t.shape).unsqueeze(-1).unsqueeze(-1)
        g_t_sq_t = diffusion._extract(diffusion.g_t_sq, t, x_t.shape)

        def get_d_log_q(score):
            term1 = torch.sum(dx * score, dim=(1, 2, 3), keepdim=True)
            drift_term = f_t - (g_t_sq_t / 2) * score
            term2 = (div_f_t_t + torch.sum(drift_term * score, dim=(1, 2, 3), keepdim=True)) * dt
            return term1 + term2

        d_log_q_shape = get_d_log_q(score_shape)
        d_log_q_color = get_d_log_q(score_color)

        log_q_shape += d_log_q_shape
        log_q_color += d_log_q_color

        x = x_prev

    return x


def generate_manifold_plot(shape_model, color_model, diffusion, config):
    """
    Generates samples and visualizes their 2D PCA projection.
    This now uses SUPERDIFF for the composed samples.
    """
    print("\n--- Generating data for manifold plot ---")
    num_samples_per_set = 150

    # --- 1. Generate Data ---
    # To get samples from individual manifolds, we can use fixed weights.
    def sample_fixed_weight(s_idx, c_idx, num, w_s, w_c):
        # A simplified sampler for fixed weights, doesn't need density tracking.
        x = torch.randn((num, 3, Config.IMG_SIZE, Config.IMG_SIZE), device=diffusion.device)
        shape_c = torch.tensor([s_idx] * num, device=diffusion.device)
        color_c = torch.tensor([c_idx] * num, device=diffusion.device)
        for i in tqdm(range(diffusion.timesteps - 1, -1, -1), desc=f"Sampling Manifold", leave=False):
            t = torch.full((num,), i, device=diffusion.device, dtype=torch.long)
            eps_s = shape_model(x, t, shape_c)
            eps_c = color_model(x, t, color_c)
            eps = w_s * eps_s + w_c * eps_c
            x = diffusion.p_sample(eps, x, t)
        return x

    print("Generating samples from the SHAPE manifold...")
    red_idx = config.COLORS.index("red")
    shape_samples = []
    for s_idx in range(len(config.SHAPES)):
        samples = sample_fixed_weight(s_idx, red_idx, num_samples_per_set // len(config.SHAPES), 1.0, 0.0)
        shape_samples.append(samples.cpu())

    print("Generating samples from the COLOR manifold...")
    circle_idx = config.SHAPES.index("circle")
    color_samples = []
    for c_idx in range(len(config.COLORS)):
        samples = sample_fixed_weight(circle_idx, c_idx, num_samples_per_set // len(config.COLORS), 0.0, 1.0)
        color_samples.append(samples.cpu())

    print("Generating samples from the COMPOSED (SUPERDIFF) distribution...")
    h_shape_idx = config.SHAPES.index(config.HOLDOUT_COMBINATION[0])
    h_color_idx = config.COLORS.index(config.HOLDOUT_COMBINATION[1])
    composed_samples = sample_superdiff(
        shape_model, color_model, diffusion,
        h_shape_idx, h_color_idx, num_images=num_samples_per_set,
        strategy='OR', temp=10.0  # High temp makes it more decisive
    ).cpu()

    # --- 2. Perform and Plot PCA ---
    all_samples = torch.cat([*shape_samples, *color_samples, composed_samples])
    all_samples_flat = all_samples.view(all_samples.size(0), -1).numpy()

    print("Performing PCA...")
    pca = PCA(n_components=2)
    pca.fit(all_samples_flat)

    shape_manifold_2d = pca.transform(torch.cat(shape_samples).view(num_samples_per_set, -1))
    color_manifold_2d = pca.transform(torch.cat(color_samples).view(num_samples_per_set, -1))
    composed_points_2d = pca.transform(composed_samples.view(num_samples_per_set, -1))

    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(shape_manifold_2d[:, 0], shape_manifold_2d[:, 1], alpha=0.6, s=60, label=f'Shape Manifold (Red Shapes)',
               c='orangered', edgecolors='w', linewidth=0.5)
    ax.scatter(color_manifold_2d[:, 0], color_manifold_2d[:, 1], alpha=0.6, s=60, label=f'Color Manifold (Circles)',
               c='dodgerblue', edgecolors='w', linewidth=0.5)
    ax.scatter(composed_points_2d[:, 0], composed_points_2d[:, 1], alpha=0.9, s=80,
               label=f'Composed ({config.HOLDOUT_COMBINATION[0]}, {config.HOLDOUT_COMBINATION[1]})', c='darkviolet',
               marker='*', edgecolors='k', linewidth=0.5)
    ax.set_title('2D PCA of Image Manifolds (SUPERDIFF)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    output_path = os.path.join(config.OUTPUT_DIR, "manifold_scatter_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved manifold plot to {output_path}")
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    # Setup
    config = Config()
    diffusion = DiffusionSDE(
        timesteps=config.TIMESTEPS,
        img_dims=(3, config.IMG_SIZE, config.IMG_SIZE),
        device=config.DEVICE
    )
    train_dataset = ShapesDataset(config, is_train=True)
    if Config.SANITY:
        dataset = tiny_subset(train_dataset, Config.SANITY_NUM_EXAMPLES)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # --- Train Models ---
    shape_model = UNet(num_classes=len(config.SHAPES)).to(config.DEVICE)
    shape_optimizer = torch.optim.Adam(shape_model.parameters(), lr=config.LR)
    train_model(shape_model, train_dataloader, diffusion, shape_optimizer, config.NUM_EPOCHS, 'shape', config)
    color_model = UNet(num_classes=len(config.COLORS)).to(config.DEVICE)
    color_optimizer = torch.optim.Adam(color_model.parameters(), lr=config.LR)
    train_model(color_model, train_dataloader, diffusion, color_optimizer, config.NUM_EPOCHS, 'color', config)

    # --- Load Best Models for Inference ---
    print("\n--- Loading best models for generation ---")
    shape_model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, "best_shape_model.pth")))
    color_model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, "best_color_model.pth")))
    shape_model.eval()
    color_model.eval()

    # --- Generate and Save Manifold Plot using SUPERDIFF ---
    generate_manifold_plot(shape_model, color_model, diffusion, config)

    # --- Generate Final Composition Grid using SUPERDIFF ---
    print("\n--- Generating final composition grid for all combinations ---")
    generated_images = []
    for s_idx, s_name in enumerate(config.SHAPES):
        for c_idx, c_name in enumerate(config.COLORS):
            print(f"Generating combination: {s_name}, {c_name}")

            generated_image = sample_superdiff(
                shape_model, color_model, diffusion, s_idx, c_idx, num_images=1,
                strategy='OR', temp=1.0  # Temperature T=1
            )
            generated_images.append(generated_image.cpu())

    grid = make_grid(torch.cat(generated_images), nrow=len(config.COLORS), normalize=True, value_range=(-1, 1))
    grid_pil = ToPILImage()(grid)
    output_path = os.path.join(config.OUTPUT_DIR, "superdiff_generation_grid.png")
    grid_pil.save(output_path)
    print(f"\nSaved final generation grid to {output_path}")

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_pil)
    plt.axis('off')
    plt.title("SUPERDIFF Compositional Generation Results", fontsize=16)
    plt.show()

