import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize, Normalize
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import math
import os


# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EXP_NAME = "shape_and_color_cross_attention"
    IMG_SIZE = 64
    BATCH_SIZE = 128
    TIMESTEPS = 300
    NUM_EPOCHS = 1  # More complex model, might need more/less tuning
    LR = 2e-4

    # --- NEW: Guidance and Composition Config ---
    GUIDANCE_DROPOUT = 0.1  # Probability of dropping a condition
    GUIDANCE_STRENGTH_SHAPE = 7.5
    GUIDANCE_STRENGTH_COLOR = 7.5

    # --- NEW: Model Config ---
    EMBED_DIM = 128  # Dimension fo embeddings and context

    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]
    HOLDOUT_COMBINATION = ("triangle", "blue")
    OUTPUT_DIR = f"diffusion_guided_unet"


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- 1. Dataset Generation ---
# Modified to provide both a grayscale input and a colored target
class ShapesDataset(Dataset):
    def __init__(self, size=5000, img_size=64, holdout=None, train=True):
        self.size = size
        self.img_size = img_size
        self.shapes = Config.SHAPES
        self.colors = Config.COLORS
        self.holdout = holdout
        self.train = train

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]
        if self.holdout and self.train:
            self.all_combinations.remove(self.holdout)

        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

        self.transform_rgb = Compose([
            Resize(img_size), CenterCrop(img_size), ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])
        self.transform_gray = Compose([
            Resize(img_size), CenterCrop(img_size), ToTensor(),
            Normalize([0.5], [0.5])  # Scale to [-1, 1]
        ])

    def __len__(self):
        return self.size

    def _draw_shape(self, shape, color_name, draw):
        margin = self.img_size // 4
        top_left, bottom_right = (margin, margin), (self.img_size - margin, self.img_size - margin)
        if shape == "circle":
            draw.ellipse([top_left, bottom_right], fill=color_name)
        elif shape == "square":
            draw.rectangle([top_left, bottom_right], fill=color_name)
        elif shape == "triangle":
            p1, p2, p3 = (self.img_size // 2, margin), (margin, self.img_size - margin), (
            self.img_size - margin, self.img_size - margin)
            draw.polygon([p1, p2, p3], fill=color_name)

    def __getitem__(self, idx):
        shape_name, color_name = self.all_combinations[idx % len(self.all_combinations)]

        # Create colored image (for loss target)
        image_rgb_pil = Image.new("RGB", (self.img_size, self.img_size), "black")
        draw_rgb = ImageDraw.Draw(image_rgb_pil)
        self._draw_shape(shape_name, color_name, draw_rgb)

        # Create grayscale image (for model input)
        image_gray_pil = Image.new("L", (self.img_size, self.img_size), "black")
        draw_gray = ImageDraw.Draw(image_gray_pil)
        self._draw_shape(shape_name, "white", draw_gray)

        shape_idx = self.shape_to_idx[shape_name]
        color_idx = self.color_to_idx[color_name]

        return self.transform_gray(image_gray_pil), self.transform_rgb(image_rgb_pil), torch.tensor(
            shape_idx), torch.tensor(color_idx)


# --- 2. Diffusion Logic (DDPM) ---
betas = torch.linspace(0.0001, 0.02, Config.TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)


def q_sample(x_start, t, noise=None):
    if noise is None: noise = torch.randn_like(x_start).to(Config.DEVICE)
    sqrt_a_t = alphas_cumprod[t.cpu()].to(Config.DEVICE).view(-1, 1, 1, 1)
    sqrt_1m_a_t = torch.sqrt(1. - alphas_cumprod[t.cpu()]).to(Config.DEVICE).view(-1, 1, 1, 1)
    return sqrt_a_t * x_start + sqrt_1m_a_t * noise


# --- 3. U-Net Model with Cross-Attention ---
# --- 3. U-Net Model with Cross-Attention ---
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
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, context):
        attn_output, _ = self.attention(x, context, context)
        return attn_output


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.attn = CrossAttention(out_channels)
        self.attn_norm = nn.LayerNorm(out_channels)

    def forward(self, x, t, context):
        h = self.norm1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)

        batch_size, channels, height, width = h.shape
        h_reshaped = h.view(batch_size, channels, -1).permute(0, 2, 1)
        attn_out = self.attn(h_reshaped, context)
        h_reshaped = h_reshaped + attn_out
        h_reshaped = self.attn_norm(h_reshaped)

        h = h_reshaped.permute(0, 2, 1).view(batch_size, channels, height, width)
        h = self.norm2(self.act(self.conv2(h)))
        return h


class GuidedUNet(nn.Module):
    def __init__(self, num_shapes=len(Config.SHAPES), num_colors=len(Config.COLORS), embed_dim=Config.EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        self.shape_embedding = nn.Embedding(num_shapes + 1, embed_dim)
        self.color_embedding = nn.Embedding(num_colors + 1, embed_dim)
        self.null_shape_idx = num_shapes
        self.null_color_idx = num_colors

        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(embed_dim), nn.Linear(embed_dim, embed_dim),
                                      nn.SiLU())

        # FIX: Input is 3-channel (noisy RGB)
        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # --- FIX: Corrected Channel Dimensions ---
        self.down1 = UNetBlock(64, 128, embed_dim)
        self.down2 = UNetBlock(128, 256, embed_dim)
        self.pool = nn.MaxPool2d(2)

        self.bot1 = UNetBlock(256, 512, embed_dim)
        self.bot2 = UNetBlock(512, 512, embed_dim)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_block1 = UNetBlock(512, 256, embed_dim)  # 256 (from up1) + 256 (from d2)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_block2 = UNetBlock(256, 128, embed_dim)  # 128 (from up2) + 128 (from d1)

        self.out_conv = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x, t, shape_labels, color_labels):
        t_emb = self.time_mlp(t)
        shape_emb = self.shape_embedding(shape_labels)
        color_emb = self.color_embedding(color_labels)
        context = torch.cat([shape_emb, color_emb], dim=1).unsqueeze(1)

        # --- FIX: Corrected Forward Pass Logic ---
        x = self.init_conv(x)

        d1 = self.down1(x, t_emb, context)
        d2 = self.down2(self.pool(d1), t_emb, context)

        b1 = self.bot1(self.pool(d2), t_emb, context)
        b2 = self.bot2(b1, t_emb, context)

        u1 = self.up1(b2)
        u2 = self.up_block1(torch.cat([u1, d2], dim=1), t_emb, context)

        u3 = self.up2(u2)
        u4 = self.up_block2(torch.cat([u3, d1], dim=1), t_emb, context)

        return self.out_conv(u4)


# --- 4. Training ---
def train(model, dataloader, optimizer, debug: bool = False):
    for epoch in range(Config.NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        for step, (gray_imgs, rgb_imgs, shapes, colors) in enumerate(progress_bar):
            optimizer.zero_grad()
            gray_imgs, rgb_imgs = gray_imgs.to(Config.DEVICE), rgb_imgs.to(Config.DEVICE)
            if debug:
                if step % dataloader.dataset.__len__():
                    from src.utils.visualization import visualize_images
                    visualize_images(gray_imgs, denormalize=False)
                    visualize_images(rgb_imgs, denormalize=False)
            shapes, colors = shapes.to(Config.DEVICE), colors.to(Config.DEVICE)

            t = torch.randint(0, Config.TIMESTEPS, (gray_imgs.shape[0],), device=Config.DEVICE)
            noise = torch.randn_like(rgb_imgs).to(Config.DEVICE)
            noisy_imgs = q_sample(rgb_imgs, t, noise)

            # Classifier-Free Guidance: Randomly drop conditions
            mask = torch.rand(shapes.shape[0], device=Config.DEVICE) > Config.GUIDANCE_DROPOUT
            shapes = torch.where(mask, shapes, model.null_shape_idx)
            mask = torch.rand(colors.shape[0], device=Config.DEVICE) > Config.GUIDANCE_DROPOUT
            colors = torch.where(mask, colors, model.null_color_idx)

            predicted_noise = model(noisy_imgs, t, shapes, colors)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        if epoch % Config.NUM_EPOCHS == 0:
            model.eval()
            generated_image = sample_image(model)
            samples_dir: Path = Path(Config.OUTPUT_DIR) / "unconditional" / f"epoch_{epoch}"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for(i, img) in enumerate(generated_image[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, samples_dir / Path(f"{Config.PREFIX}_{i:03d}.png"), normalize=False)

# --- 5. Sampling (with Composition) ---
@torch.no_grad()
def sample_image(model):
    model.eval()
    x = torch.randn((1, 3, Config.IMG_SIZE, Config.IMG_SIZE), device=Config.DEVICE)

    uncond_shape_labels = torch.tensor([model.null_shape_idx], device=Config.DEVICE)
    uncond_color_labels = torch.tensor([model.null_color_idx], device=Config.DEVICE)

    for i in tqdm(reversed(range(Config.TIMESTEPS)), desc="Diffusion Sampling", leave=False):
        t = torch.full((1,), i, device=Config.DEVICE)

        pred_uncond = model(x, t, uncond_shape_labels, uncond_color_labels)

        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]

        coeff_img = 1 / torch.sqrt(alpha_t)
        coeff_pred_noise = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
        x =  coeff_img * (x -coeff_pred_noise  * pred_uncond)
        if i > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(betas[i]) * noise

    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def sample_composed(model, shape_idx, color_idx):
    model.eval()
    x = torch.randn((1, 3, Config.IMG_SIZE, Config.IMG_SIZE), device=Config.DEVICE)

    shape_labels = torch.tensor([shape_idx], device=Config.DEVICE)
    color_labels = torch.tensor([color_idx], device=Config.DEVICE)
    uncond_shape_labels = torch.tensor([model.null_shape_idx], device=Config.DEVICE)
    uncond_color_labels = torch.tensor([model.null_color_idx], device=Config.DEVICE)

    for i in tqdm(reversed(range(Config.TIMESTEPS)), desc="Composed Sampling", leave=False):
        t = torch.full((1,), i, device=Config.DEVICE)

        # Predict noise for all 4 conditions
        pred_shape_only = model(x, t, shape_labels, uncond_color_labels)
        pred_color_only = model(x, t, uncond_shape_labels, color_labels)
        pred_uncond = model(x, t, uncond_shape_labels, uncond_color_labels)

        # Perform score interpolation
        shape_guidance = pred_shape_only - pred_uncond
        color_guidance = pred_color_only - pred_uncond
        final_pred_noise = pred_uncond + Config.GUIDANCE_STRENGTH_SHAPE * shape_guidance + Config.GUIDANCE_STRENGTH_COLOR * color_guidance

        # Denoise step
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        x = 1 / torch.sqrt(alpha_t) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * final_pred_noise)
        if i > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(betas[i]) * noise

    return (x.clamp(-1, 1) + 1) / 2


# --- Main Execution ---
if __name__ == '__main__':
    model = GuidedUNet().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    dataset = ShapesDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    print("--- Training Guided Diffusion Model with Cross-Attention ---")
    train(model, dataloader, optimizer)

    print("\n--- Generating Final Composition Grid ---")
    generated_images = []
    for s_idx, s_name in enumerate(Config.SHAPES):
        for c_idx, c_name in enumerate(Config.COLORS):
            print(f"Generating Shape: {s_name}, Color: {c_name}")
            img = sample_composed(model, s_idx, c_idx)
            generated_images.append(img.cpu())

    grid = make_grid(torch.cat(generated_images), nrow=len(Config.COLORS))
    output_path = os.path.join(Config.OUTPUT_DIR, "guided_composition_grid.png")
    save_image(grid, output_path)
    print(f"\n🎉 Final composition grid saved to {output_path}!")
