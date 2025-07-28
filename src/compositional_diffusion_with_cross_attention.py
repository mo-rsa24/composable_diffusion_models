import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Normalize
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import math

from src.utils.tools import tiny_subset, CheckpointManager, is_cluster


# --- Configuration ---
class Configuration:
    def __init__(self, exp_name: str, run:str , results_dir: Path, ckpt_mgr: CheckpointManager = None, lr: float = 2e-4, sanity: bool=False, device: str = 'cuda'):
        self.DEVICE = device
        self.sanity = sanity
        self.sanity_num_examples = 8
        self.ckpt_mgr = ckpt_mgr
        self.exp_name = exp_name
        self.run=run

        self.IMG_SIZE = 32
        self.BATCH_SIZE = 4 if self.sanity else 128
        self.TIMESTEPS = 500

        self.NUM_EPOCHS = 1 if self.sanity else 400
        self.LOG_EVERY_EPOCH = 1 if self.sanity else 20
        self.LR = lr

        # Guidance and Composition Config
        self.GUIDANCE_DROPOUT = 0.1  # Probability of dropping a condition during training
        self.GUIDANCE_STRENGTH_SHAPE = 7.5
        self.GUIDANCE_STRENGTH_COLOR = 7.5

        # Model Config
        self.EMBED_DIM = 128  # Dimension for embeddings and context
        self.OUTPUT_DIR: Path = results_dir


# --- 1. Dataset ---
# We use grayscale MNIST and provide the color as a separate label.
class GrayscaleColorDataset(Dataset):
    def __init__(self, Config: Configuration, train=True):
        self.mnist_dataset = MNIST(root="./data", train=train, download=True,
                                   transform=Resize((Config.IMG_SIZE, Config.IMG_SIZE)))
        self.colors = ['red', 'green', 'blue']
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Get grayscale image and normalize to [-1, 1]
        img, label = self.mnist_dataset[idx]
        img_tensor = ToTensor()(img)  # To [0, 1]
        img_tensor = Normalize((0.5,), (0.5,))(img_tensor)  # To [-1, 1]

        # Assign a color label. Can be random or based on digit.
        # Random assignment forces more robust learning.
        color_idx = torch.randint(0, len(self.colors), (1,)).item()
        img_tensor = img_tensor.repeat(3, 1, 1)
        return img_tensor, torch.tensor(label), torch.tensor(color_idx)


# --- 2. Diffusion Model with Cross-Attention ---

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
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4):
        super().__init__()
        # Use kdim and vdim to specify the dimensions of the key and value tensors (the context).
        # The query's dimension is specified by embed_dim.
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=context_dim,
            vdim=context_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x, context):
        # x: (batch, seq_len, query_dim) - image features
        # context: (batch, context_len, context_dim) - guidance embeddings
        attn_output, _ = self.attention(query=x, key=context, value=context)
        return attn_output

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        # Cross-Attention Layer
        self.attn = CrossAttention(query_dim=out_channels, context_dim=context_dim)
        self.attn_norm = nn.LayerNorm(out_channels)

    def forward(self, x, t, context):
        h = self.norm1(self.conv1(x))

        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)

        h = self.act(h)

        # Reshape for attention
        batch_size, channels, height, width = h.shape
        h_reshaped = h.view(batch_size, channels, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Apply Cross-Attention
        attn_out = self.attn(h_reshaped, context)
        h_reshaped = h_reshaped + attn_out  # Residual connection
        h_reshaped = self.attn_norm(h_reshaped)

        # Reshape back
        h = h_reshaped.permute(0, 2, 1).view(batch_size, channels, height, width)

        h = self.norm2(self.conv2(h))
        return self.act(h)


class GuidedUNet(nn.Module):
    def __init__(self, num_digits=10, num_colors=3, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

        # Embeddings for guidance
        self.digit_embedding = nn.Embedding(num_digits + 1, embed_dim)  # +1 for null
        self.color_embedding = nn.Embedding(num_colors + 1, embed_dim)  # +1 for null
        self.null_digit_idx = num_digits
        self.null_color_idx = num_colors

        # U-Net Architecture
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(embed_dim), nn.Linear(embed_dim, embed_dim),
                                      nn.SiLU())

        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Context dimension is twice the embed_dim (digit + color)
        context_dim = embed_dim * 2

        self.down1 = UNetBlock(64, 128, embed_dim, context_dim)
        self.down2 = UNetBlock(128, 256, embed_dim, context_dim)
        self.pool = nn.MaxPool2d(2)

        self.bot1 = UNetBlock(256, 512, embed_dim, context_dim)
        self.bot2 = UNetBlock(512, 256, embed_dim, context_dim)

        # FIXED [1]: Input to up1 is b2, which has 256 channels.
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up2 = UNetBlock(256 + 128, 128, embed_dim, context_dim)

        # FIXED [2]: Input to up3 is u2, which has 128 channels.
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up4 = UNetBlock(128 + 64, 64, embed_dim, context_dim)

        # FIXED [3]: Input is cat(u4, x) which is 64 + 64 = 128 channels.
        # The 'x' variable is the one processed by init_conv.
        self.out_conv = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x, t, digit_labels, color_labels):
        t_emb = self.time_mlp(t)

        # Create context vector from embeddings
        digit_emb = self.digit_embedding(digit_labels)
        color_emb = self.color_embedding(color_labels)
        context = torch.cat([digit_emb, color_emb], dim=1).unsqueeze(1)  # (B, 1, 2*embed_dim)

        x = self.init_conv(x)  # x shape becomes (B, 64, H, W)

        d1 = self.down1(x, t_emb, context)
        d2 = self.down2(self.pool(d1), t_emb, context)

        b1 = self.bot1(self.pool(d2), t_emb, context)
        b2 = self.bot2(b1, t_emb, context)  # b2 shape is (B, 256, H/4, W/4)

        u1 = self.up1(b2)  # u1 shape is (B, 128, H/2, W/2)
        u2 = self.up2(torch.cat([u1, d2], dim=1), t_emb, context)  # u2 shape is (B, 128, H/2, W/2)

        u3 = self.up3(u2)  # u3 shape is (B, 64, H, W)
        u4 = self.up4(torch.cat([u3, d1], dim=1), t_emb, context)  # u4 shape is (B, 64, H, W)

        # Final concatenation uses the 64-channel 'x' from after init_conv
        final_in = torch.cat([u4, x], dim=1)  # Shape is (B, 128, H, W)

        return self.out_conv(final_in)


# --- 3. Diffusion Logic ---
def get_coeff(Config: Configuration, beta_start: float = 0.0001, beta_end:float =0.02):
    betas = torch.linspace(beta_start, beta_end, Config.TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(Config.DEVICE)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(Config.DEVICE)
    return alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

def q_sample(Config: Configuration, x_start, t, noise=None):
    if noise is None: noise = torch.randn_like(x_start)
    _, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_coeff(Config)
    sqrt_a_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(Config.DEVICE)
    sqrt_1m_a_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(Config.DEVICE)
    return sqrt_a_t * x_start + sqrt_1m_a_t * noise


# --- 4. Training ---
def train(Config: Configuration, model, dataloader, optimizer):
    for epoch in range(Config.NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        for imgs, digits, colors in progress_bar:
            optimizer.zero_grad()
            imgs = imgs.to(Config.DEVICE)
            digits = digits.to(Config.DEVICE)
            colors = colors.to(Config.DEVICE)

            t = torch.randint(0, Config.TIMESTEPS, (imgs.shape[0],), device=Config.DEVICE)

            # Create the target colored image for loss calculation
            # This is a "pseudo-target" as the model learns to generate color from grayscale
            color_map = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]], device=Config.DEVICE)
            target_colors = color_map[colors]
            target_rgb = imgs * target_colors.view(-1, 3, 1, 1)

            noise = torch.randn_like(target_rgb)
            x_noisy = q_sample(Config, imgs, t, torch.randn_like(imgs))  # Noise is added to grayscale

            # Classifier-Free Guidance: Randomly drop conditions
            mask = torch.rand(digits.shape[0], device=Config.DEVICE) > Config.GUIDANCE_DROPOUT
            digits = torch.where(mask, digits, model.null_digit_idx)
            mask = torch.rand(colors.shape[0], device=Config.DEVICE) > Config.GUIDANCE_DROPOUT
            colors = torch.where(mask, colors, model.null_color_idx)

            predicted_noise_or_img = model(x_noisy, t, digits, colors)
            # (torch.Size([4, 1, 32, 32]), torch.Size([4]), torch.Size([4]), torch.Size([4]))
            # Loss is calculated on the colored target
            loss = F.mse_loss(predicted_noise_or_img, target_rgb)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())


@torch.no_grad()
def sample_composed(Config: Configuration, model, digit, color_idx):
    model.eval()

    # Start with noise (1 channel, as input is grayscale)
    x = torch.randn((1, 3, Config.IMG_SIZE, Config.IMG_SIZE), device=Config.DEVICE)

    # Prepare labels for all guidance scenarios
    digit_labels = torch.tensor([digit], device=Config.DEVICE)
    color_labels = torch.tensor([color_idx], device=Config.DEVICE)
    uncond_digit_labels = torch.tensor([model.null_digit_idx], device=Config.DEVICE)
    uncond_color_labels = torch.tensor([model.null_color_idx], device=Config.DEVICE)

    for i in tqdm(reversed(range(Config.TIMESTEPS)), desc="Composed Sampling", leave=False):
        t = torch.full((1,), i, device=Config.DEVICE)

        # Predict noise for all 4 conditions in a single batch
        # This is highly efficient
        x_batch = x.repeat(4, 1, 1, 1)
        t_batch = t.repeat(4)

        digits_batch = torch.cat([digit_labels, digit_labels, uncond_digit_labels, uncond_digit_labels])
        colors_batch = torch.cat([color_labels, uncond_color_labels, color_labels, uncond_color_labels])

        pred_batch = model(x_batch, t_batch, digits_batch, colors_batch)

        pred_full, pred_shape_only, pred_color_only, pred_uncond = torch.chunk(pred_batch, 4)

        # Perform score interpolation (Classifier-Free Guidance formula adapted for two conditions)
        shape_guidance = pred_shape_only - pred_uncond
        color_guidance = pred_color_only - pred_uncond

        # The final prediction is the unconditional prediction plus the scaled guidance from each expert
        final_pred = pred_uncond + Config.GUIDANCE_STRENGTH_SHAPE * shape_guidance + Config.GUIDANCE_STRENGTH_COLOR * color_guidance

        # Denoise step (using DDIM for faster sampling)
        alphas_cumprod, _, __ = get_coeff(Config)
        alpha_t = alphas_cumprod[i]
        alpha_t_prev = alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)

        # Predict x0
        pred_x0 = final_pred

        # Get direction to x0
        dir_xt = torch.sqrt(1. - alpha_t_prev) * final_pred

        # Get x_{t-1}
        x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

    return (x.clamp(-1, 1) + 1) / 2  # Denormalize to [0, 1]


# --- Main Execution ---
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


    Config = Configuration(args.exp_name, args.run_id, results_dir, ckpt_mgr, lr=args.lr, sanity=args.sanity, device=device)



    model = GuidedUNet(embed_dim=Config.EMBED_DIM).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    dataset = GrayscaleColorDataset(Config, train=True)
    if Config.sanity:
        dataset = tiny_subset(dataset, Config.sanity_num_examples)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    print("--- Training Guided Diffusion Model with Cross-Attention ---")
    train(Config, model, dataloader, optimizer)

    print("\n--- Generating Final Composition Grid ---")
    generated_images = []
    for digit in range(10):
        for color_idx in range(3):
            print(f"Generating Digit: {digit}, Color: {['Red', 'Green', 'Blue'][color_idx]}")
            img = sample_composed(Config, model, digit, color_idx)
            generated_images.append(img.cpu())

    grid = make_grid(torch.cat(generated_images), nrow=3)
    output_path = Config.OUTPUT_DIR /  "guided_composition_grid.png"
    save_image(grid, output_path)
    print(f"\n🎉 Final composition grid saved to {output_path}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose two diffusion models.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--run_id", type=str, required=True, help="A unique ID for the current run.")
    parser.add_argument("--project_name", type=str, default="mini-composable-diffusion-model",
                        help="Name of the project directory.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)
