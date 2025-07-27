import os
from torchvision.utils import save_image
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid
from src.diffusion.samplers import SuperDiffSampler
torch.manual_seed(0)
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def visualize_training_epoch(model, cfg, sde, device, results_dir, model_name, epoch):
    model.eval()
    sampler = SuperDiffSampler(sde)
    shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)
    print("Generating samples from Model A (Content)...")
    samples = sampler.sample_single_model(model, cfg.sampling.batch_size, shape, device)

    path_a: Path = results_dir / model_name / f"epoch_{epoch}"
    path_a.mkdir(parents=True, exist_ok=True)

    print(f"Saved Model A samples to {path_a}")
    for (i, img) in enumerate(samples[:4]):
        img = img.detach().cpu().clamp(-1, 1)
        img = (img + 1) / 2  # map [-1,1] -> [0,1]
        save_image(img, path_a / Path(f"{cfg.experiment.name}_epoch_{epoch}_sample_{i:03d}.png"), normalize=False)


def visualize_images(
    data,
    num_images: int = 4,
    denormalize: bool = True,
    mean: list = [0.5],
    std: list = [0.5],
    save_path: str = None,
    title: str = None,
    show: bool = True,
):
    # --- Convert input to tensor ---
    if isinstance(data, Image.Image):
        data = torch.from_numpy(np.array(data)).unsqueeze(-1).permute(2, 0,
                                                                      1).float() / 255 if data.mode == 'L' else torch.from_numpy(
            np.array(data)).permute(2, 0, 1).float() / 255
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = data.clone().detach()  # Avoid modifying original

    # --- Normalize shape ---
    if data.ndim == 2:
        data = data.unsqueeze(0)  # (H, W) → (1, H, W)

    if data.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        if data.shape[0] <= 4:  # (C, H, W)
            batch = data.unsqueeze(0)  # (1, C, H, W)
        else:  # (H, W, C)
            data = data.permute(2, 0, 1)
            batch = data.unsqueeze(0)
    elif data.ndim == 4:
        batch = data
    else:
        raise ValueError(f"Unsupported input shape: {data.shape}")

    # --- Slice if needed ---
    batch = batch[:num_images]

    # --- De-normalize if required ---
    if denormalize:
        mean = torch.tensor(mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(std, device=batch.device).view(-1, 1, 1)
        batch = batch * std + mean

    # --- Create grid ---
    grid_img = make_grid(batch, nrow=min(num_images, int(np.sqrt(num_images))), normalize=False)

    # --- Convert for plotting ---
    img = grid_img.permute(1, 2, 0).cpu().numpy()

    # Grayscale
    if img.shape[2] == 1:
        img = img.squeeze(-1)
        cmap = "gray"
    else:
        cmap = None

    # --- Plot ---
    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(img, cmap=cmap)

    # --- Save or Show ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
