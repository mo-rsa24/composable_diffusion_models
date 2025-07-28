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
    VAE_EPOCHS = 25
    VAE_BETA = 4.0

    # Latent Diffusion Config
    LDM_EPOCHS = 100
    TIMESTEPS = 300

    # --- MODIFICATION 1: Define holdout combinations ---
    # We will explicitly EXCLUDE these from the training set.
    # The model will never see a Blue 7 or a Red 1.
    HOLDOUT_COMBINATIONS = [
        {'digit': 7, 'color_idx': 2},  # Blue 7
        {'digit': 1, 'color_idx': 0},  # Red 1
    ]

    LR = 1e-3
    OUTPUT_DIR = "latent_diffusion_composition_output_part_3"


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- 1. VAE Model and Dataset ---
# --- MODIFICATION 2: Rewrite the dataset to exclude holdout combinations ---
class ColoredMNIST(Dataset):
    def __init__(self, train=True):
        mnist_dataset = MNIST(root="./data", train=train, download=True)
        self.transforms = Resize((Config.IMG_SIZE, Config.IMG_SIZE), antialias=True)
        # Colors: 0=Red, 1=Green, 2=Blue
        self.color_map = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])

        # We build our dataset manually to exclude holdouts
        self.data = []
        self.digit_targets = []
        self.color_targets = []

        holdouts = {(c['digit'], c['color_idx']) for c in Config.HOLDOUT_COMBINATIONS}
        print(f"Building dataset and excluding holdout combinations: {holdouts}...")

        for img, label in tqdm(zip(mnist_dataset.data, mnist_dataset.targets), total=len(mnist_dataset),
                               desc="Filtering Dataset"):
            digit = label.item()

            # Assign a random color, but ensure it's not a holdout combination
            while True:
                color_idx = np.random.randint(0, 3)  # Randomly pick Red, Green, or Blue
                if (digit, color_idx) not in holdouts:
                    break  # We found a valid color, so we keep it

            self.data.append(img)
            self.digit_targets.append(label)
            self.color_targets.append(torch.tensor(color_idx))

        # Convert lists to tensors for efficiency
        self.data = torch.stack(self.data)
        self.digit_targets = torch.stack(self.digit_targets)
        self.color_targets = torch.stack(self.color_targets)
        print("Dataset built successfully.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].float() / 255.0
        digit_label = self.digit_targets[idx]
        color_label = self.color_targets[idx]  # Use the pre-assigned color

        color = self.color_map[color_label]
        # Create the 3-channel colored image
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

    def encode(self, x):
        result = self.encoder(x)
        return self.fc_mu(result), self.fc_log_var(result)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# --- 2. Latent Dataset ---
class LatentDataset(Dataset):
    def __init__(self, vae_model, original_dataset):
        super().__init__()
        self.latents = []
        self.digit_labels = []
        self.color_labels = []

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

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.digit_labels[idx], self.color_labels[idx]


# --- 3. Latent Diffusion Model (MLP) ---
class LatentDiffusionMLP(nn.Module):
    def __init__(self, latent_dim, num_classes, time_emb_dim=32):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # In LatentDiffusionMLP
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 2 * time_emb_dim, 512), nn.ReLU(),  # Wider
            nn.Linear(512, 1024), nn.ReLU(),  # Wider and Deeper
            nn.Linear(1024, 1024), nn.ReLU(),  # Deeper
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, t, y_digit, y_color):
        t_emb = self.time_mlp(t.float().unsqueeze(1) / Config.TIMESTEPS)
        digit_emb = self.label_emb(y_digit)
        color_emb = self.label_emb(y_color)  # Note: This model architecture has been simplified for clarity
        # A more robust approach might use separate embeddings for each condition.
        # Here we use a single embedding layer and rely on the training to distinguish.
        # For this to work, we need a single MLP class and instantiate two of them.
        x_combined = torch.cat([x, t_emb, digit_emb, color_emb], dim=-1)
        return self.model(x_combined)


class ComposableLatentDiffusionMLP(nn.Module):
    """A dedicated MLP for a single condition (shape or color)."""

    def __init__(self, latent_dim, num_classes, time_emb_dim=32):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # The model input combines the latent vector, time embedding, and label embedding
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


# --- 4. Diffusion Logic (in Latent Space) ---
betas = torch.linspace(0.0001, 0.02, Config.TIMESTEPS, device=Config.DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


def q_sample_latent(z_start, t, noise=None):
    if noise is None: noise = torch.randn_like(z_start)
    # Get correct shape for broadcasting
    sqrt_a_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_1m_a_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    return sqrt_a_t * z_start + sqrt_1m_a_t * noise


def train_latent_diffusion(model, dataloader, optimizer, num_epochs, condition_type):
    print(f"--- Training Latent Diffusion Expert: {condition_type.upper()} ---")
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for z, digits, colors in progress_bar:
            optimizer.zero_grad()
            z = z.to(Config.DEVICE)
            # Select the correct label based on the expert being trained
            labels = (digits if condition_type == 'shape' else colors).to(Config.DEVICE)

            t = torch.randint(0, Config.TIMESTEPS, (z.shape[0],), device=Config.DEVICE).long()
            noise = torch.randn_like(z)
            z_noisy = q_sample_latent(z, t, noise)

            predicted_noise = model(z_noisy, t, labels)
            loss = F.mse_loss(noise, predicted_noise)

            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())


# --- 5. Compositional Sampling ---
@torch.no_grad()
def sample_composed_latent(shape_expert, color_expert, vae_decoder, shape_idx, color_idx, w_shape=1.5, w_color=0.7):
    shape_expert.eval()
    color_expert.eval()

    z = torch.randn((1, Config.VAE_LATENT_DIMS), device=Config.DEVICE)
    shape_label = torch.tensor([shape_idx], device=Config.DEVICE)
    color_label = torch.tensor([color_idx], device=Config.DEVICE)

    for i in tqdm(reversed(range(Config.TIMESTEPS)), desc="Composed Latent Sampling", total=Config.TIMESTEPS,
                  leave=False):
        t = torch.full((1,), i, device=Config.DEVICE)

        # Get noise predictions from both experts
        pred_noise_shape = shape_expert(z, t, shape_label)
        pred_noise_color = color_expert(z, t, color_label)

        # Compose the noise predictions via weighted average
        composed_noise = (w_shape * pred_noise_shape + w_color * pred_noise_color) / (w_shape + w_color)

        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]

        # Denoise step (DDPM sampling formula)
        z = (1 / torch.sqrt(alpha_t)) * (z - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * composed_noise)
        if i > 0:
            noise = torch.randn_like(z)
            z += torch.sqrt(betas[i]) * noise

    # Decode the final latent vector into an image
    return vae_decoder(z).squeeze(0)


# --- Main Execution ---
if __name__ == '__main__':
    # --- Part 1: Train the VAE ---
    print("--- Training VAE Encoder/Decoder ---")
    vae = BetaVAE(latent_dims=Config.VAE_LATENT_DIMS).to(Config.DEVICE)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=Config.LR)

    # Use the modified dataset that excludes holdouts
    full_dataset = ColoredMNIST(train=True)
    vae_dataloader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Simple VAE training loop
    for epoch in range(1, Config.VAE_EPOCHS + 1):
        vae.train()
        loop = tqdm(vae_dataloader, desc=f"VAE Epoch [{epoch}/{Config.VAE_EPOCHS}]", leave=True)
        total_loss = 0
        for data, _, _ in loop:
            data = data.to(Config.DEVICE)
            vae_optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)

            # ELBO loss = Reconstruction Loss + KL Divergence
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + Config.VAE_BETA * kld

            loss.backward()
            vae_optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item() / len(data))

    # --- Part 2: Create Latent Dataset ---
    latent_dataset = LatentDataset(vae, full_dataset)
    latent_dataloader = DataLoader(latent_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # --- Part 3: Train Latent Diffusion Experts ---
    # We need two separate models, one for shape and one for color
    shape_expert = ComposableLatentDiffusionMLP(Config.VAE_LATENT_DIMS, num_classes=10).to(Config.DEVICE)
    color_expert = ComposableLatentDiffusionMLP(Config.VAE_LATENT_DIMS, num_classes=3).to(Config.DEVICE)

    shape_optimizer = torch.optim.Adam(shape_expert.parameters(), lr=Config.LR)
    color_optimizer = torch.optim.Adam(color_expert.parameters(), lr=Config.LR)

    train_latent_diffusion(shape_expert, latent_dataloader, shape_optimizer, Config.LDM_EPOCHS, 'shape')
    train_latent_diffusion(color_expert, latent_dataloader, color_optimizer, Config.LDM_EPOCHS, 'color')

    # --- MODIFICATION 3: Generate and save the specific holdout images first ---
    print("\n--- Generating Holdout Combination Samples ---")
    for combo in Config.HOLDOUT_COMBINATIONS:
        digit = combo['digit']
        color_idx = combo['color_idx']
        color_name = ['Red', 'Green', 'Blue'][color_idx]

        print(f"Generating Held-Out Combo -> Digit: {digit}, Color: {color_name}")
        img = sample_composed_latent(shape_expert, color_expert, vae.decode, digit, color_idx)

        output_path = os.path.join(Config.OUTPUT_DIR, f"holdout_digit_{digit}_color_{color_name}.png")
        save_image(img.cpu(), output_path)
        print(f"-> Saved to {output_path}")

    # --- Part 4: Perform Compositional Sampling for a full grid ---
    print("\n--- Generating Final Composition Grid ---")
    generated_images = []
    # Generate a 10x3 grid of all combinations
    for digit in range(10):
        row_images = []
        for color_idx in range(3):
            print(f"Generating Digit: {digit}, Color: {['Red', 'Green', 'Blue'][color_idx]}")
            img = sample_composed_latent(shape_expert, color_expert, vae.decode, digit, color_idx)
            row_images.append(img.cpu())
        generated_images.extend(row_images)

    grid = make_grid(torch.stack(generated_images), nrow=3)  # 3 columns for 3 colors
    output_path = os.path.join(Config.OUTPUT_DIR, "final_composition_grid.png")
    save_image(grid, output_path)
    print(f"\n🎉 Final composition grid saved to {output_path}!")
    print("Check the output directory for the generated holdout images and the full grid.")