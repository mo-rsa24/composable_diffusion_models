import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
IMG_SIZE = 32
TIMESTEPS = 300
LEARNING_RATE = 1e-3
EPOCHS = 10  # Increase for better quality, 10 is enough for a demo
MODEL_A_PATH = "model_green_2.pth"
MODEL_B_PATH = "model_red_6.pth"


# --- 1. Data Preparation ---

def get_colored_mnist_subset(digit, color_channel, name):
    """
    Filters the MNIST dataset for a specific digit and applies a color.
    Color channels: 0 for Red, 1 for Green, 2 for Blue.
    """
    print(f"Preparing dataset: Colored '{name}'")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Filter for the specific digit
    indices = [i for i, (img, label) in enumerate(full_dataset) if label == digit]
    subset = Subset(full_dataset, indices)

    # Create a new dataset with colored images
    colored_data = []
    for img, label in subset:
        # Create a 3-channel image
        colored_img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        colored_img[color_channel] = img[0]  # Put the digit in the specified color channel
        colored_data.append((colored_img, label))

    return colored_data


# --- 2. U-Net Model Architecture ---
# A simple U-Net is sufficient for MNIST.

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
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bn2(self.relu(self.conv2(h)))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=DEVICE) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    def __init__(self):
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

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
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


# --- 3. Diffusion Process ---

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# Forward Process
betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def forward_diffusion_sample(x_0, t, device=DEVICE):
    """ Takes an image and a timestep and returns the noisy version of it """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(
        device), noise.to(device)


# --- 4. Training ---

def train_model(model_name, dataset):
    print(f"\n--- Training Model for: {model_name} ---")
    model = SimpleUnet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader)
        for step, (batch, _) in enumerate(pbar):
            optimizer.zero_grad()

            t = torch.randint(0, TIMESTEPS, (batch.shape[0],), device=DEVICE).long()
            x_noisy, noise = forward_diffusion_sample(batch, t, DEVICE)
            predicted_noise = model(x_noisy, t)

            loss = nn.functional.l1_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Model saved to {model_name}.pth")
    return model


# --- 5. Sampling and Composition ---

@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise and returns the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t.all() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_composed(model_a, model_b, weight_a=1.0, weight_b=1.0):
    """
    Samples from the combined distribution of two models.
    This is the core of the superposition logic.
    """
    print("\n--- Sampling from Composed Models ---")
    img_shape = (1, 3, IMG_SIZE, IMG_SIZE)
    img = torch.randn(img_shape, device=DEVICE)

    pbar = tqdm(reversed(range(0, TIMESTEPS)), total=TIMESTEPS, desc="Composing")
    for i in pbar:
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)

        # Predict noise from each model
        noise_a = model_a(img, t)
        noise_b = model_b(img, t)

        # Combine the noise predictions (the "score" superposition)
        combined_noise = (weight_a * noise_a + weight_b * noise_b) / (weight_a + weight_b)

        # Manually perform the denoising step using the combined noise
        betas_t = get_index_from_list(betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, img.shape)

        model_mean = sqrt_recip_alphas_t * (img - betas_t * combined_noise / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(posterior_variance, t, img.shape)

        if i == 0:
            img = model_mean
        else:
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    return img


def show_image(image_tensor, title=""):
    """Helper to display a tensor as an image."""
    # Reverse the normalization
    image = image_tensor.detach().cpu().squeeze(0)
    image = torch.clamp(image, 0, 1)  # Clamp values to be in the [0, 1] range
    plt.imshow(image.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Create Datasets
    green_2_dataset = get_colored_mnist_subset(digit=2, color_channel=1, name="Green 2")
    red_6_dataset = get_colored_mnist_subset(digit=6, color_channel=0, name="Red 6")

    # 2. Train Models (or load if they exist)
    try:
        model_a = SimpleUnet().to(DEVICE)
        model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
        print(f"Loaded pre-trained model for Green 2 from {MODEL_A_PATH}")
    except FileNotFoundError:
        model_a = train_model("model_green_2", green_2_dataset)

    try:
        model_b = SimpleUnet().to(DEVICE)
        model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
        print(f"Loaded pre-trained model for Red 6 from {MODEL_B_PATH}")
    except FileNotFoundError:
        model_b = train_model("model_red_6", red_6_dataset)

    # 3. Compose the models during sampling
    # The weights allow you to control the influence of each model.
    # Equal weights mean we want an equal blend of "green 2-ness" and "red 6-ness".
    composed_image = sample_composed(model_a, model_b, weight_a=1.0, weight_b=1.0)

    # 4. Show the result
    show_image(composed_image, "Composed: Green 2 + Red 6")