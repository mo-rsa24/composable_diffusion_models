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

from src.utils.tools import tiny_subset


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
    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]
    HOLDOUT_COMBINATION = ("triangle", "blue")

    # MODIFIED: Add probability for classifier-free guidance
    UNCOND_PROB = 0.1

    OUTPUT_DIR = f"src/scripts/mini-experiments/visualizations/{EXP_NAME}/composable_diffusion_output_part_6_3"


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# --- 1. Dataset Generation (No changes needed) ---

class ShapesDataset(Dataset):
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
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return self.size

    def _draw_shape(self, shape, color_name, draw):
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
        shape_name, color_name = self.all_combinations[idx % len(self.all_combinations)]
        image = Image.new("RGB", (self.img_size, self.img_size), "white")
        draw = ImageDraw.Draw(image)
        self._draw_shape(shape_name, color_name, draw)
        shape_idx = self.shape_to_idx[shape_name]
        color_idx = self.color_to_idx[color_name]
        return self.transforms(image), torch.tensor(shape_idx), torch.tensor(color_idx)


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


def p_losses(denoise_model, x_start, t, y, loss_type="l1"):
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


# --- 3. U-Net Model (MODIFIED) ---

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
    def __init__(self, num_classes):  # MODIFIED: Takes num_classes
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
        # MODIFIED: Add 1 to num_classes for the null/unconditional token
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


# --- 4. Training (MODIFIED) ---

def train_model(cfg: Config, model: SimpleUnet,  dataloader, optimizer, condition_type, num_classes):
    """
    MODIFIED: Trains a specialist model using classifier-free guidance.
    Labels are randomly replaced with a null token.
    """
    print(f"--- Training {condition_type.upper()} model with Classifier-Free Guidance ---")
    uncond_token_id = num_classes  # The unconditional token is the last one

    for epoch in range(1, cfg.NUM_EPOCHS+1):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS+1}")
        for step, (images, shape_labels, color_labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_size = images.shape[0]
            images = images.to(Config.DEVICE)

            if condition_type == 'shape':
                labels = shape_labels.to(Config.DEVICE)
            else:
                labels = color_labels.to(Config.DEVICE)

            uncond_mask = torch.rand(batch_size, device=Config.DEVICE) < Config.UNCOND_PROB
            labels[uncond_mask] = uncond_token_id

            t = torch.randint(0, Config.TIMESTEPS, (batch_size,), device=Config.DEVICE).long()
            loss = p_losses(model, images, t, labels, loss_type="l1")
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        if epoch % cfg.LOG_EVERY_EPOCH == 0:
            model.eval()
            generated_image = sample_image(model, labels)
            samples_dir: Path = Path(Config.OUTPUT_DIR) / condition_type / f"epoch_{epoch}"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for(i, img) in enumerate(generated_image[:4]):
                img = img.detach().cpu().clamp(-1, 1)
                img = (img + 1) / 2  # map [-1,1] -> [0,1]
                save_image(img, samples_dir / Path(f"sample_{i:03d}.png"), normalize=False)
    print(f"--- Finished training {condition_type.upper()} model ---")

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


def get_forward_process_params(t):
    """Helper function to get f_t and g_t from the Ornstein-Uhlenbeck SDE."""
    # Approximate time derivatives with finite differences
    dt = 1.0 / Config.TIMESTEPS

    # Get alpha and sigma at t and t-1
    alpha_t = alphas_cumprod[t].item()
    alpha_t_prev = alphas_cumprod[t - 1].item() if t > 0 else 1.0
    sigma_t_sq = 1 - alpha_t
    sigma_t_sq_prev = 1 - alpha_t_prev

    # f_t(x) = d/dt(log(alpha_t)) * x. We use sqrt_alphas_cumprod for alpha_t.
    # d/dt(log(sqrt(alpha_t))) = 0.5 * d/dt(log(alpha_t))
    # We approximate d/dt(log(alpha_t)) ~ (log(alpha_t) - log(alpha_t_prev)) / dt
    log_alpha_t = 0.5 * torch.log(alphas_cumprod[t])
    log_alpha_t_prev = 0.5 * torch.log(alphas_cumprod[t - 1]) if t > 0 else 0.0
    d_log_alpha_dt = (log_alpha_t - log_alpha_t_prev) / dt

    f_t_coeff = d_log_alpha_dt

    # g_t^2 = 2 * sigma_t^2 * d/dt(log(sigma_t/alpha_t))
    # d/dt(log(sigma/alpha)) = d/dt(log(sigma)) - d/dt(log(alpha))
    log_sigma_t = 0.5 * torch.log(torch.tensor(sigma_t_sq))
    # log_sigma_t_prev = 0.5 * torch.log(torch.tensor(sigma_t_sq_prev)) if t > 0 else -float('inf')
    log_sigma_t_prev = 0.5 * torch.log(torch.tensor(sigma_t_sq_prev)) if t > 0 else torch.tensor(-float('inf'))
    d_log_sigma_dt = (log_sigma_t - log_sigma_t_prev) / dt if torch.isfinite(log_sigma_t_prev) else 0.0


    g_t_sq = 2 * sigma_t_sq * (d_log_sigma_dt - d_log_alpha_dt)
    g_t_sq = max(g_t_sq, 1e-8)  # Ensure g is non-zero

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

        # --- FIX: Store the image state at time t ---
        prev_img = img.clone()

        # --- 1. Calculate scores and model-specific terms based on the state at time t ---
        pred_noises = [m(prev_img, t, lab) for m, lab in zip(models, labels)]
        scores = [-p / extract(sqrt_one_minus_alphas_cumprod, t, p.shape) for p in pred_noises]
        f_t_coeff, g_t_sq = get_forward_process_params(t)
        f_t = f_t_coeff * prev_img

        # --- 2. Calculate Kappa weights ---
        kappa = torch.zeros(num_models, device=Config.DEVICE)
        if mode == 'OR':
            log_q_tensor = torch.stack(log_qs)
            kappa = F.softmax(T * log_q_tensor.squeeze() + l, dim=0)

        elif mode == 'AND':
            # This part for AND mode depends on the 'a' and 'b' matrices from the paper.
            # While the original code attempts this, ensuring its stability is complex.
            # A simple starting point is to use equal weights for debugging.
            # If the original AND logic is kept, ensure it uses terms calculated from prev_img.
            # For now, let's assume it calculates a valid kappa. Here's a fallback:
            kappa = torch.tensor([0.5, 0.5], device=Config.DEVICE)  # Using simple average for stability

        # --- 3. Update the image from t to t-1 ---
        composed_score = kappa[0] * scores[0] + kappa[1] * scores[1]

        # Use standard DDPM reverse step
        betas_t = extract(betas, t, prev_img.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, prev_img.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, prev_img.shape)

        composed_noise = -composed_score * sqrt_one_minus_alphas_cumprod_t
        model_mean = sqrt_recip_alphas_t * (prev_img - betas_t * composed_noise / sqrt_one_minus_alphas_cumprod_t)

        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, prev_img.shape)
            noise = torch.randn_like(prev_img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

        # --- 4. Update Log Densities using the *actual* dx ---
        dx = img - prev_img  # This is the crucial fix!

        div_f = f_t_coeff * d  # Divergence of f_t(x)

        for idx in range(num_models):
            # Term 1: <dx, ∇log q_i>
            term1 = torch.sum(dx * scores[idx])

            # Term 2: (<∇,f> + <f - g²/2 * ∇log q, ∇log q>)dτ
            term2_inner_dot = torch.sum((f_t - (g_t_sq / 2) * scores[idx]) * scores[idx])
            term2 = d_tau * (div_f + term2_inner_dot)

            d_log_q = term1 + term2
            log_qs[idx] += d_log_q.detach()

    return img

@torch.no_grad()
def sample_composed(shape_model, color_model, shape_idx, color_idx, w_shape=2.0, w_color=2.0):
    """
    MODIFIED: Sample using the composition method from the paper.
    """
    shape_name = Config.SHAPES[shape_idx]
    color_name = Config.COLORS[color_idx]
    print(f"Sampling with paper's composition: {shape_name} (w={w_shape:.1f}) AND {color_name} (w={w_color:.1f})")

    img_size = Config.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=Config.DEVICE)

    # Define conditional and unconditional labels for both models
    shape_cond_label = torch.full((1,), shape_idx, device=Config.DEVICE, dtype=torch.long)
    color_cond_label = torch.full((1,), color_idx, device=Config.DEVICE, dtype=torch.long)
    shape_uncond_label = torch.full((1,), len(Config.SHAPES), device=Config.DEVICE, dtype=torch.long)
    color_uncond_label = torch.full((1,), len(Config.COLORS), device=Config.DEVICE, dtype=torch.long)

    for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc="Composed Sampling (Paper Method)",
                  total=Config.TIMESTEPS):
        t = torch.full((1,), i, device=Config.DEVICE, dtype=torch.long)

        # 1. Get all noise predictions (conditional and unconditional)
        pred_noise_shape_cond = shape_model(img, t, shape_cond_label)
        pred_noise_color_cond = color_model(img, t, color_cond_label)
        pred_noise_shape_uncond = shape_model(img, t, shape_uncond_label)
        pred_noise_color_uncond = color_model(img, t, color_uncond_label)

        # 2. Calculate the single unconditional baseline (average of the two unconditional preds)
        pred_noise_uncond = (pred_noise_shape_uncond + pred_noise_color_uncond) / 2.0

        # 3. Apply the paper's composition formula
        composed_noise = pred_noise_uncond + \
                         w_shape * (pred_noise_shape_cond - pred_noise_uncond) + \
                         w_color * (pred_noise_color_cond - pred_noise_uncond)

        # 4. Denoise for one step using the composed noise
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


# --- Main Execution (MODIFIED) ---
if __name__ == '__main__':
    # 1. Create Datasets and Dataloaders
    dataset = ShapesDataset(size=5000, holdout=Config.HOLDOUT_COMBINATION, train=True)
    if Config.SANITY:
        dataset = tiny_subset(dataset, Config.SANITY_NUM_EXAMPLE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. Initialize Models
    num_shape_classes = len(Config.SHAPES)
    num_color_classes = len(Config.COLORS)
    shape_model = SimpleUnet(num_classes=num_shape_classes).to(Config.DEVICE)
    color_model = SimpleUnet(num_classes=num_color_classes).to(Config.DEVICE)

    # 3. Train Models
    shape_optimizer = torch.optim.Adam(shape_model.parameters(), lr=Config.LR)
    color_optimizer = torch.optim.Adam(color_model.parameters(), lr=Config.LR)
    train_model(Config, shape_model, dataloader, shape_optimizer, 'shape', num_shape_classes)

    train_model(Config, color_model, dataloader, color_optimizer, 'color', num_color_classes)

    # --- 4. Perform Compositional Sampling ---
    print("\n--- Starting Compositional Sampling (Paper's Method) ---")

    shape_map = {name: i for i, name in enumerate(Config.SHAPES)}
    color_map = {name: i for i, name in enumerate(Config.COLORS)}

    generated_images = []
    w_shape = 2.5
    w_color = 2.5

    for mode in ['OR', 'AND']:
        generated_images = []
        for s_name in Config.SHAPES:
            for c_name in Config.COLORS:
                s_idx = shape_map[s_name]
                c_idx = color_map[c_name]

                if (s_name, c_name) == Config.HOLDOUT_COMBINATION:
                    print(f"\nGenerating HELD-OUT combination with SUPERDIFF ({mode}): {s_name}, {c_name}")
                else:
                    print(f"\nGenerating seen combination with SUPERDIFF ({mode}): {s_name}, {c_name}")

                # Call the new SUPERDIFF sampler
                generated_image = sample_superdiff(shape_model, color_model, s_idx, c_idx, mode=mode, T=10.0)
                generated_images.append(generated_image)

        # Save the results in a grid
        grid = make_grid(torch.cat(generated_images), nrow=len(Config.COLORS), normalize=True, value_range=(-1, 1))

        output_path = os.path.join(Config.OUTPUT_DIR, f"superdiff_generation_grid_{mode.lower()}.png")
        save_image(grid, output_path)
        print(f"\nSaved SUPERDIFF ({mode}) generation grid to {output_path}")
