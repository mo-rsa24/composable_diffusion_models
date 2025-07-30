import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from pathlib import Path
from tqdm import tqdm
import math
import os

from src.utils.tools import tiny_subset


# --- Configuration ---
# --- Configuration (MODIFIED) ---
class Config:
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
   EXP_NAME = "shape_and_color_paper_method"  # Changed experiment name
   SANITY = False
   SANITY_NUM_EXAMPLE = 8
   IMG_SIZE = 64
   BATCH_SIZE = 4 if SANITY else 128
   PREFIX = "samples"
   TIMESTEPS = 500
   NUM_EPOCHS = 1 if SANITY else 200
   LOG_EVERY_EPOCH = 1 if SANITY else 20
   LR = 1e-4
   UNCOND_PROB = 0.1

   OUTPUT_DIR = f"src/scripts/mini-experiments/visualizations/{EXP_NAME}/composable_diffusion_output_part_7"




# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- 1. Dataset for Digits (Using ColoredMNIST) ---

class ColoredMNIST(Dataset):
    """
    A wrapper for MNIST that filters by digit and applies a specific color.
    This version is simplified to use one color for all target digits.
    """

    def __init__(self, image_size, target_digits, color_name, color_rgb):
        self.image_size = image_size
        self.target_digits = target_digits
        self.color_name = color_name
        self.color_rgb_tensor = torch.tensor(color_rgb).view(3, 1, 1)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # Load MNIST and filter for target digits
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True)
        self.indices = [i for i, (_, label) in enumerate(mnist_dataset) if label in self.target_digits]
        self.data = [mnist_dataset[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image_tensor = self.transform(image)

        # Make the image 3-channel and apply color
        # The digit is white (1.0) on a black background (0.0)
        colored_image = image_tensor.repeat(3, 1, 1) * self.color_rgb_tensor

        # Normalize to [-1, 1]
        final_image = (colored_image * 2) - 1

        return final_image, label


# --- 2. Diffusion Logic (DDPM - No changes needed) ---

def linear_beta_schedule(timesteps, device='cpu'):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


betas = linear_beta_schedule(timesteps=Config.TIMESTEPS, device=Config.DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, labels, loss_type="l1"):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, labels)
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss


# --- 3. U-Net Model ---

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
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.gn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t_emb))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.gn2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUnet(nn.Module):
    def __init__(self, num_classes=1):
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
        self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)
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


# --- 4. Training and Sampling ---

def get_forward_process_params(t):
    """Helper function to get f_t and g_t^2 for the SDE."""
    dt = 1.0 / Config.TIMESTEPS
    log_alpha_t = 0.5 * torch.log(alphas_cumprod[t])
    log_alpha_t_prev = 0.5 * torch.log(alphas_cumprod[t - 1]) if t > 0 else 0.0
    d_log_alpha_dt = (log_alpha_t - log_alpha_t_prev) / dt
    f_t_coeff = d_log_alpha_dt

    sigma_t_sq = 1 - alphas_cumprod[t].item()
    sigma_t_sq_prev = 1 - (alphas_cumprod[t - 1].item() if t > 0 else 1.0)
    log_sigma_t = 0.5 * torch.log(torch.tensor(sigma_t_sq)) if sigma_t_sq > 0 else torch.tensor(-float('inf'))
    log_sigma_t_prev = 0.5 * torch.log(torch.tensor(sigma_t_sq_prev)) if sigma_t_sq_prev > 0 else torch.tensor(
        -float('inf'))
    d_log_sigma_dt = (log_sigma_t - log_sigma_t_prev) / dt if torch.isfinite(log_sigma_t_prev) and torch.isfinite(
        log_sigma_t) else 0.0

    g_t_sq = 2 * sigma_t_sq * (d_log_sigma_dt - d_log_alpha_dt)
    g_t_sq = max(g_t_sq, 1e-8)
    return f_t_coeff, g_t_sq


@torch.no_grad()
def sample_superdiff(model1, model2, label1_idx, label2_idx, mode='OR', T=1.0, l=0.0):
    """
    Sample using the SUPERDIFF algorithm on two generic models.
    """
    print(f"Sampling with SUPERDIFF ({mode}) on two models.")

    img_size = Config.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=Config.DEVICE)
    d = 3 * img_size * img_size

    log_q1 = torch.zeros(1, device=Config.DEVICE)
    log_q2 = torch.zeros(1, device=Config.DEVICE)
    cond_label1 = torch.tensor([label1_idx], device=Config.DEVICE, dtype=torch.long)
    cond_label2 = torch.tensor([label2_idx], device=Config.DEVICE, dtype=torch.long)

    models = [model1, model2]
    labels = [cond_label1, cond_label2]
    log_qs = [log_q1, log_q2]
    num_models = len(models)

    for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc=f"SUPERDIFF ({mode}) Sampling", total=Config.TIMESTEPS):
        t = torch.full((1,), i, device=Config.DEVICE, dtype=torch.long)
        d_tau = 1.0 / Config.TIMESTEPS

        kappa = torch.zeros(num_models, device=Config.DEVICE)
        pred_noises = [m(img, t, lab) for m, lab in zip(models, labels)]
        scores = [-p / extract(sqrt_one_minus_alphas_cumprod, t, p.shape) for p in pred_noises]

        if mode == 'OR':
            log_q_tensor = torch.stack(log_qs)
            kappa = F.softmax(T * log_q_tensor.squeeze() + l, dim=0)

        elif mode == 'AND':
            f_t_coeff, g_t_sq = get_forward_process_params(t)
            f_t = f_t_coeff * img
            g_t = torch.sqrt(torch.tensor(g_t_sq, device=Config.DEVICE))
            div_f = f_t_coeff * d
            reverse_drifts = [-f_t + (g_t_sq / 2) * s for s in scores]

            a = torch.zeros(num_models, num_models, device=Config.DEVICE)
            for r in range(num_models):
                for c in range(num_models):
                    a[r, c] = d_tau * torch.sum(reverse_drifts[c] * scores[r])

            dW_noise = torch.randn_like(img) * torch.sqrt(torch.tensor(d_tau, device=Config.DEVICE))
            b = torch.zeros(num_models, device=Config.DEVICE)
            for r in range(num_models):
                deterministic_part = d_tau * (div_f + torch.sum((f_t - (g_t_sq / 2) * scores[r]) * scores[r]))
                stochastic_part = torch.sum(g_t * dW_noise * scores[r])
                b[r] = deterministic_part + stochastic_part

            A_mat = torch.tensor([
                [(a[0, 0] - a[1, 0]).item(), (a[0, 1] - a[1, 1]).item()],
                [1.0, 1.0]
            ], device=Config.DEVICE)
            B_vec = torch.tensor([(b[1] - b[0]).item(), 1.0], device=Config.DEVICE)

            try:
                kappa = torch.linalg.solve(A_mat, B_vec)
                kappa = torch.clamp(kappa, min=0, max=1.0)
                kappa /= torch.sum(kappa)
            except torch.linalg.LinAlgError:
                kappa = torch.tensor([0.5, 0.5], device=Config.DEVICE)
        else:
            raise ValueError("Mode must be 'OR' or 'AND'")

        composed_score = kappa[0] * scores[0] + kappa[1] * scores[1]
        betas_t = extract(betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, img.shape)
        composed_noise = -composed_score * sqrt_one_minus_alphas_cumprod_t
        model_mean = sqrt_recip_alphas_t * (img - betas_t * composed_noise / sqrt_one_minus_alphas_cumprod_t)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, img.shape)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

        f_t_coeff, _ = get_forward_process_params(t)
        f_t = f_t_coeff * img
        div_f = f_t_coeff * d
        # This line was missing g_t definition for the update step
        f_t_coeff, g_t_sq = get_forward_process_params(t)
        g_t = torch.sqrt(torch.tensor(g_t_sq, device=Config.DEVICE))
        reverse_drifts = [-f_t + (g_t_sq / 2) * s for s in scores]
        composed_drift = kappa[0] * reverse_drifts[0] + kappa[1] * reverse_drifts[1]
        # This line was missing dW_noise definition for the update step
        dW_noise = torch.randn_like(img) * torch.sqrt(torch.tensor(d_tau, device=Config.DEVICE))
        dx = composed_drift * d_tau + g_t * dW_noise

        for idx in range(num_models):
            term1 = torch.sum(dx * scores[idx])
            term2_inner = torch.sum((f_t - (g_t_sq / 2) * scores[idx]) * scores[idx])
            term2 = d_tau * (div_f + term2_inner)
            d_log_q = term1 + term2
            log_qs[idx] += d_log_q.detach()

    return img


@torch.no_grad()
def sample(model, num_images=16, w=1.5):
    model.eval()
    img_size = Config.IMG_SIZE
    img = torch.randn((num_images, 3, img_size, img_size), device=Config.DEVICE)
    cond_label = torch.full((num_images,), 0, device=Config.DEVICE, dtype=torch.long)
    uncond_label = torch.full((num_images,), 1, device=Config.DEVICE, dtype=torch.long)

    for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc="Sampling", total=Config.TIMESTEPS):
        t = torch.full((num_images,), i, device=Config.DEVICE, dtype=torch.long)
        pred_noise_cond = model(img, t, cond_label)
        pred_noise_uncond = model(img, t, uncond_label)
        guided_noise = pred_noise_uncond + w * (pred_noise_cond - pred_noise_uncond)
        betas_t = extract(betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, img.shape)
        model_mean = sqrt_recip_alphas_t * (img - betas_t * guided_noise / sqrt_one_minus_alphas_cumprod_t)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, img.shape)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    model.train()
    return img


def train_model(cfg: Config, model: SimpleUnet, dataloader, optimizer, model_name: str):
    """
    Trains a single specialist model.
    """
    print(f"--- Training '{model_name}' model with Classifier-Free Guidance ---")
    uncond_token_id = 1

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}")
        for step, (images, original_labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_size = images.shape[0]
            images = images.to(Config.DEVICE)

            # *** FIX APPLIED HERE ***
            # Create labels for the specialist model: all are class 0 (the concept).
            # The original_labels from ColoredMNIST (e.g., 6) are ignored,
            # preventing the out-of-bounds error.
            labels = torch.zeros(batch_size, device=Config.DEVICE, dtype=torch.long)

            # Randomly replace some labels with the unconditional token
            uncond_mask = torch.rand(batch_size, device=Config.DEVICE) < cfg.UNCOND_PROB
            labels[uncond_mask] = uncond_token_id

            t = torch.randint(0, cfg.TIMESTEPS, (batch_size,), device=Config.DEVICE).long()
            loss = p_losses(model, images, t, labels, loss_type="l1")
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        if epoch % cfg.LOG_EVERY_EPOCH == 0 or epoch == cfg.NUM_EPOCHS:
            print(f"Epoch {epoch}: Logging samples...")
            generated_images = sample(model, num_images=16, w=1.5)
            generated_images = (generated_images.clamp(-1, 1) + 1) / 2
            grid = make_grid(generated_images, nrow=4)
            # Create subdirectory for the model's samples
            model_sample_dir = Path(cfg.OUTPUT_DIR) / model_name
            model_sample_dir.mkdir(exist_ok=True)
            save_path = model_sample_dir / f"epoch_{epoch}.png"
            save_image(grid, save_path)
            print(f"Saved sample grid to {save_path}")

    print(f"--- Finished training '{model_name}' model ---")
    torch.save(model.state_dict(), Path(cfg.OUTPUT_DIR) / f"{model_name}_final.pth")


# --- Main Execution ---
if __name__ == '__main__':
    # --- Train Red 6 Model ---
    print("Initializing dataset for RED 6s...")
    red_6_dataset = ColoredMNIST(image_size=Config.IMG_SIZE, target_digits=[6], color_name='red',
                                 color_rgb=(1.0, 0.0, 0.0))
    if Config.SANITY:
        red_6_dataset = tiny_subset(red_6_dataset, Config.SANITY_NUM_EXAMPLE)
    red_6_dataloader = DataLoader(red_6_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    red_6_model = SimpleUnet(num_classes=1).to(Config.DEVICE)
    red_6_optimizer = torch.optim.Adam(red_6_model.parameters(), lr=Config.LR)
    train_model(Config, red_6_model, red_6_dataloader, red_6_optimizer, 'red_6_model')

    # --- Train Green 2 Model ---
    print("\n" + "=" * 50 + "\n")
    print("Initializing dataset for GREEN 2s...")
    green_2_dataset = ColoredMNIST(image_size=Config.IMG_SIZE, target_digits=[2], color_name='green',
                                   color_rgb=(0.0, 1.0, 0.0))
    if Config.SANITY:
        green_2_dataset = tiny_subset(green_2_dataset, Config.SANITY_NUM_EXAMPLE)
    green_2_dataloader = DataLoader(green_2_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    green_2_model = SimpleUnet(num_classes=1).to(Config.DEVICE)
    green_2_optimizer = torch.optim.Adam(green_2_model.parameters(), lr=Config.LR)
    train_model(Config, green_2_model, green_2_dataloader, green_2_optimizer, 'green_2_model')

    # --- Perform SuperDiff Compositional Sampling ---
    print("\n" + "=" * 50 + "\n")
    print("--- Starting SUPERDIFF Compositional Sampling ---")

    label_idx = 0
    num_samples_per_mode = 8

    for mode in ['OR', 'AND']:
        generated_images = []
        print(f"\nGenerating {num_samples_per_mode} samples with SUPERDIFF ({mode})...")
        for i in range(num_samples_per_mode):
            generated_image = sample_superdiff(
                red_6_model,
                green_2_model,
                label_idx,
                label_idx,
                mode=mode,
                T=10.0
            )
            generated_images.append(generated_image)

        grid = make_grid(torch.cat(generated_images), nrow=4, normalize=True, value_range=(-1, 1))
        output_path = Path(Config.OUTPUT_DIR) / f"superdiff_composition_{mode.lower()}.png"
        save_image(grid, output_path)
        print(f"Saved SUPERDIFF ({mode}) generation grid to {output_path}")

    print("\nAll training and sampling complete.")
