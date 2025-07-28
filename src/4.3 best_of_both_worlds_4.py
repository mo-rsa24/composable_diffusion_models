import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 32
    BATCH_SIZE = 256

    # VAE Config
    VAE_LATENT_DIMS = 10
    VAE_EPOCHS = 30  # Increased for better latent space
    VAE_BETA = 8.0  # Increased for better disentanglement

    # Latent Diffusion Config
    LDM_EPOCHS = 100  # Increased for better expert training
    TIMESTEPS = 300

    # --- MODIFICATION 1: Add CFG Parameters ---
    # Probability of using an unconditional token during training
    P_UNCOND = 0.15
    # Guidance strength for each expert during sampling
    CFG_SCALE_SHAPE = 7.5
    CFG_SCALE_COLOR = 8.0

    HOLDOUT_COMBINATIONS = [
        {'digit': 7, 'color_idx': 2},  # Blue 7
        {'digit': 1, 'color_idx': 0},  # Red 1
    ]

    LR = 1e-3
    OUTPUT_DIR = "latent_diffusion_composition_output_part_4"


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- VAE Model and Dataset (No Changes Here) ---
class ColoredMNIST(Dataset):
    def __init__(self, train=True):
        mnist_dataset = MNIST(root="./data", train=train, download=True)
        self.transforms = Resize((Config.IMG_SIZE, Config.IMG_SIZE), antialias=True)
        self.color_map = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])  # Red, Green, Blue

        self.data = []
        self.digit_targets = []
        self.color_targets = []

        holdouts = {(c['digit'], c['color_idx']) for c in Config.HOLDOUT_COMBINATIONS}
        print(f"Building dataset and excluding holdout combinations: {holdouts}...")

        for img, label in tqdm(zip(mnist_dataset.data, mnist_dataset.targets), total=len(mnist_dataset),
                               desc="Filtering Dataset"):
            digit = label.item()

            while True:
                color_idx = np.random.randint(0, 3)
                if (digit, color_idx) not in holdouts:
                    break

            self.data.append(img)
            self.digit_targets.append(label)
            self.color_targets.append(torch.tensor(color_idx))

        self.data = torch.stack(self.data)
        self.digit_targets = torch.stack(self.digit_targets)
        self.color_targets = torch.stack(self.color_targets)
        print("Dataset built successfully.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].float() / 255.0
        digit_label = self.digit_targets[idx]
        color_label = self.color_targets[idx]

        color = self.color_map[color_label]
        img_rgb = torch.stack([img * color[0], img * color[1], img * color[2]])
        return self.transforms(img_rgb), digit_label, color_label


class BetaVAE(nn.Module):
    def __init__(self, latent_dims):
        super(BetaVAE, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dims)
        self.fc_log_var = nn.Linear(256, latent_dims)
        self.decoder_input = nn.Linear(latent_dims, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x): return self.fc_mu(self.encoder(x)), self.fc_log_var(self.encoder(x))

    def reparameterize(self, mu, log_var): return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def decode(self, z): return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class LatentDataset(Dataset):
    def __init__(self, vae_model, original_dataset):
        super().__init__()
        self.latents, self.digit_labels, self.color_labels = [], [], []
        vae_model.eval()
        dataloader = DataLoader(original_dataset, batch_size=Config.BATCH_SIZE)
        with torch.no_grad():
            for imgs, digits, colors in tqdm(dataloader, desc="Creating Latent Dataset"):
                mu, _ = vae_model.encode(imgs.to(Config.DEVICE))
                self.latents.append(mu.cpu())
                self.digit_labels.append(digits)
                self.color_labels.append(colors)
        self.latents = torch.cat(self.latents, dim=0)
        self.digit_labels = torch.cat(self.digit_labels, dim=0)
        self.color_labels = torch.cat(self.color_labels, dim=0)

    def __len__(self): return len(self.latents)

    def __getitem__(self, idx): return self.latents[idx], self.digit_labels[idx], self.color_labels[idx]


# --- MODIFICATION 2: Latent Diffusion Model with CFG in mind ---
class ComposableLatentDiffusionMLP(nn.Module):
    def __init__(self, latent_dim, num_classes, time_emb_dim=32):
        super().__init__()
        # The embedding layer has `num_classes + 1` entries.
        # The last entry is reserved for the unconditional token.
        self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        self.time_mlp = nn.Sequential(nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 2 * time_emb_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t.float().unsqueeze(1) / Config.TIMESTEPS)
        y_emb = self.label_emb(y)
        x_combined = torch.cat([x, t_emb, y_emb], dim=-1)
        return self.model(x_combined)


# --- Diffusion Logic ---
betas = torch.linspace(0.0001, 0.02, Config.TIMESTEPS, device=Config.DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


def q_sample_latent(z_start, t, noise=None):
    if noise is None: noise = torch.randn_like(z_start)
    return sqrt_alphas_cumprod[t].view(-1, 1) * z_start + sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise


# --- MODIFICATION 3: Rewritten Training loop for CFG ---
def train_latent_diffusion(model, dataloader, optimizer, num_epochs, condition_type, uncond_token):
    print(f"--- Training Latent Diffusion Expert: {condition_type.upper()} ---")
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for z, digits, colors in progress_bar:
            optimizer.zero_grad()
            z = z.to(Config.DEVICE)
            labels = (digits if condition_type == 'shape' else colors).to(Config.DEVICE)

            # With probability P_UNCOND, set the label to the unconditional token
            uncond_mask = torch.rand(labels.shape[0], device=Config.DEVICE) < Config.P_UNCOND
            labels[uncond_mask] = uncond_token

            t = torch.randint(0, Config.TIMESTEPS, (z.shape[0],), device=Config.DEVICE).long()
            noise = torch.randn_like(z)
            z_noisy = q_sample_latent(z, t, noise)

            predicted_noise = model(z_noisy, t, labels)
            loss = F.mse_loss(noise, predicted_noise)

            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())


# --- MODIFICATION 4: Rewritten Sampling loop for CFG ---
@torch.no_grad()
def sample_composed_latent_cfg(shape_expert, color_expert, vae_decoder, shape_idx, color_idx, shape_uncond_token,
                               color_uncond_token):
    shape_expert.eval()
    color_expert.eval()

    z = torch.randn((1, Config.VAE_LATENT_DIMS), device=Config.DEVICE)

    for i in tqdm(reversed(range(Config.TIMESTEPS)), desc="CFG Composed Sampling", total=Config.TIMESTEPS, leave=False):
        t = torch.full((1,), i, device=Config.DEVICE)

        # === Shape Guidance ===
        # Duplicate z for batched prediction (conditional and unconditional)
        z_batch = z.repeat(2, 1)
        t_batch = t.repeat(2)
        shape_labels = torch.tensor([shape_idx, shape_uncond_token], device=Config.DEVICE)

        # Get both predictions in one pass
        pred_noise_shape_both = shape_expert(z_batch, t_batch, shape_labels)
        pred_noise_shape_cond, pred_noise_shape_uncond = pred_noise_shape_both.chunk(2)

        # Apply CFG formula
        guided_noise_shape = pred_noise_shape_uncond + Config.CFG_SCALE_SHAPE * (
                    pred_noise_shape_cond - pred_noise_shape_uncond)

        # === Color Guidance ===
        color_labels = torch.tensor([color_idx, color_uncond_token], device=Config.DEVICE)
        pred_noise_color_both = color_expert(z_batch, t_batch, color_labels)
        pred_noise_color_cond, pred_noise_color_uncond = pred_noise_color_both.chunk(2)

        # Apply CFG formula
        guided_noise_color = pred_noise_color_uncond + Config.CFG_SCALE_COLOR * (
                    pred_noise_color_cond - pred_noise_color_uncond)

        # === Combine Guided Predictions ===
        # Average the final guided noise from both experts
        composed_noise = (guided_noise_shape + guided_noise_color) / 2.0

        # Denoise step (DDPM formula)
        alpha_t = alphas[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
        z = (1 / torch.sqrt(alpha_t)) * (z - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * composed_noise)
        if i > 0:
            noise = torch.randn_like(z)
            z += torch.sqrt(betas[i]) * noise

    return vae_decoder(z).squeeze(0)


# --- Main Execution ---
if __name__ == '__main__':
    # Part 1: Train VAE
    print("--- Training VAE Encoder/Decoder ---")
    vae = BetaVAE(latent_dims=Config.VAE_LATENT_DIMS).to(Config.DEVICE)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=Config.LR)
    full_dataset = ColoredMNIST(train=True)
    vae_dataloader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    for epoch in range(1, Config.VAE_EPOCHS + 1):
        vae.train()
        loop = tqdm(vae_dataloader, desc=f"VAE Epoch [{epoch}/{Config.VAE_EPOCHS}]", leave=True)
        for data, _, _ in loop:
            data = data.to(Config.DEVICE)
            vae_optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + Config.VAE_BETA * kld
            loss.backward()
            vae_optimizer.step()
            loop.set_postfix(loss=loss.item() / len(data))

    # Part 2: Create Latent Dataset
    latent_dataset = LatentDataset(vae, full_dataset)
    latent_dataloader = DataLoader(latent_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Part 3: Train Latent Diffusion Experts with CFG
    # Add 1 to num_classes for the unconditional token
    shape_expert = ComposableLatentDiffusionMLP(Config.VAE_LATENT_DIMS, num_classes=10).to(Config.DEVICE)
    color_expert = ComposableLatentDiffusionMLP(Config.VAE_LATENT_DIMS, num_classes=3).to(Config.DEVICE)
    shape_optimizer = torch.optim.Adam(shape_expert.parameters(), lr=Config.LR)
    color_optimizer = torch.optim.Adam(color_expert.parameters(), lr=Config.LR)

    # The unconditional token is the last index in the embedding table
    SHAPE_UNCOND_TOKEN = 10
    COLOR_UNCOND_TOKEN = 3

    train_latent_diffusion(shape_expert, latent_dataloader, shape_optimizer, Config.LDM_EPOCHS, 'shape',
                           SHAPE_UNCOND_TOKEN)
    train_latent_diffusion(color_expert, latent_dataloader, color_optimizer, Config.LDM_EPOCHS, 'color',
                           COLOR_UNCOND_TOKEN)

    # Part 4: Perform Compositional Sampling using CFG
    print("\n--- Generating Holdout Combination Samples using CFG ---")
    for combo in Config.HOLDOUT_COMBINATIONS:
        digit, color_idx = combo['digit'], combo['color_idx']
        color_name = ['Red', 'Green', 'Blue'][color_idx]
        print(f"Generating Held-Out Combo -> Digit: {digit}, Color: {color_name}")
        img = sample_composed_latent_cfg(shape_expert, color_expert, vae.decode, digit, color_idx, SHAPE_UNCOND_TOKEN,
                                         COLOR_UNCOND_TOKEN)
        save_image(img.cpu(), os.path.join(Config.OUTPUT_DIR, f"holdout_digit_{digit}_color_{color_name}.png"))

    print("\n--- Generating Final Composition Grid using CFG ---")
    generated_images = []
    for digit in range(10):
        for color_idx in range(3):
            print(f"Generating Digit: {digit}, Color: {['Red', 'Green', 'Blue'][color_idx]}")
            img = sample_composed_latent_cfg(shape_expert, color_expert, vae.decode, digit, color_idx,
                                             SHAPE_UNCOND_TOKEN, COLOR_UNCOND_TOKEN)
            generated_images.append(img.cpu())

    grid = make_grid(torch.stack(generated_images), nrow=10)  # 10 columns for 10 digits
    save_image(grid, os.path.join(Config.OUTPUT_DIR, "final_composition_grid.png"))
    print(f"\n🎉 Final CFG composition grid saved to {Config.OUTPUT_DIR}!")