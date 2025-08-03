# train_image.py - Core Logic
import torch
import torch.optim as optim
from tqdm import trange
from models.unet_small import UNet
from dataset import get_mnist_dataloader, sample_data
from schedule import q_t, sigma, alpha
from utils import set_seed, save_checkpoint
from viz import save_grid, plot_loss

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
epochs = 50
lr = 1e-4
batch_size = 128
model_path = "checkpoints/image_model.pth"

# --- Setup ---
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
dataloader = get_mnist_dataloader(batch_size=batch_size)
data_generator = sample_data(dataloader)
losses = []

# --- Training Loop ---
for epoch in range(epochs):
    for i in trange(len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}"):
        optimizer.zero_grad()

        x0 = next(data_generator).to(device)
        t = torch.rand(x0.shape[0], device=device) * (1.0 - 1e-3) + 1e-3

        xt, eps = q_t(x0, t)
        eps_hat = model(xt, t)

        loss = ((eps - eps_hat) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # --- Validation & Checkpointing ---
    if epoch % 10 == 0:
        with torch.no_grad():
            val_noise = torch.randn(64, 1, 28, 28, device=device)
            # Simple one-step denoising for a quick check
            t_val = torch.ones(64, device=device) * 0.9
            xt_val, _ = q_t(val_noise, t_val)
            eps_hat_val = model(xt_val, t_val)
            x0_hat = (xt_val - sigma(t_val).view(-1, 1, 1, 1) * eps_hat_val) / alpha(t_val).view(-1, 1, 1, 1)
            save_grid(x0_hat, f"outputs/epoch_{epoch + 1}_val.png")

        save_checkpoint(model, optimizer, epoch, model_path)
        plot_loss(losses, "outputs/image_loss.png")