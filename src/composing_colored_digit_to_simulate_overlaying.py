"""
The intent behind this code is to explore a more flexible, scalable, and interpretable alternative to traditional generative models. By breaking down a complex scene into simpler components, this approach allows for:

Dynamic Scene Generation: Users can create novel compositions on-the-fly without retraining any models.

Improved Controllability: It provides explicit, pixel-level control over where objects appear.

Efficiency: Training smaller, specialized models can be faster and require less data than training a single model to understand all possible objects and layouts.

"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from box import Box
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

from src.models.composing_colored_digit_to_simulate_overlaying import VPSDE, ScoreModel
from src.utils.tools import is_cluster, CheckpointManager, save_config_to_yaml, tiny_subset
from src.utils.visualization import visualize_training_epoch




# ==============================================================================
# 2. CUSTOM DATASET (Unchanged)
# ==============================================================================
class ColoredMNIST(Dataset):
    def __init__(self, image_size, target_digits=None, color_override=None):
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        self.indices = [i for i, (_, label) in enumerate(mnist) if target_digits is None or label in target_digits]
        self.mnist_dataset = mnist;
        self.color_override = color_override
        self.color_map = {0: (.5, .5, .5), 1: (0, .5, 1), 2: (0, .8, 0), 3: (0, .8, .8), 4: (1, .5, 0), 5: (1, 1, 0),
                          6: (1, 0, 0), 7: (1, 0, 1), 8: (.5, 0, 1), 9: (.6, .3, .1)}

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[self.indices[idx]];
        image_tensor = self.transform(image)
        color = self.color_override if self.color_override is not None else self.color_map[label]
        colored_image = image_tensor.repeat(3, 1, 1) * torch.tensor(color).view(3, 1, 1)
        return (colored_image * 2) - 1, label


# ==============================================================================
# 3. LAYOUTDIFF SAMPLER AND HELPERS
# ==============================================================================
class LayoutDiff:
    """A diffusion sampler that composes models based on spatial masks."""

    def __init__(self, sde: VPSDE):
        self.sde = sde

    @torch.no_grad()
    def sample(self, models: list[nn.Module], masks: list[torch.Tensor], shape: tuple, device: str) -> torch.Tensor:
        if len(models) != len(masks):
            raise ValueError("The number of models and masks must be equal.")

        x = torch.randn(shape, device=device)

        # Pre-calculate the final, non-overlapping mask for each model's score.
        # The last model in the list is treated as being on top.
        final_masks = [torch.zeros_like(m) for m in masks]
        occlusion_mask = torch.zeros_like(masks[0])  # Keeps track of area already claimed
        for i in range(len(masks) - 1, -1, -1):
            # The unique region is the mask's area minus what's already covered by models on top.
            unique_region = torch.clamp(masks[i] - occlusion_mask, 0, 1)
            final_masks[i] = unique_region
            occlusion_mask += unique_region

        # Add batch and channel dimensions for broadcasting and move to device
        final_masks = [m.to(device).unsqueeze(0).unsqueeze(0) for m in final_masks]

        # Standard reverse-time diffusion loop
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc="Layout-Aware Sampling", leave=False):
            t_idx = timesteps[i]
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            # --- Spatially-Aware Score Composition ---
            # Get score from each model and apply it only in its designated region.
            combined_noise_pred = torch.zeros_like(x)
            for model, mask in zip(models, final_masks):
                model.eval()
                # The model predicts the noise, which is proportional to the score.
                noise_pred = model(x, t.float())
                combined_noise_pred += noise_pred * mask

            # --- Standard DDPM-style Reverse Step ---
            # This is equivalent to the reverse SDE step but using the DDPM formulation
            # which is simpler to implement from the existing VPSDE class.
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_bar_t = self.sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)

            # Get the predicted x0 and then the mean of the posterior q(x_{t-1} | x_t, x_0)
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * combined_noise_pred) / sqrt_alpha_bar_t
            pred_x0 = torch.clamp(pred_x0, -1., 1.)

            beta_t = self.sde.betas[t].view(-1, 1, 1, 1)
            alpha_bar_prev = self.sde.alphas_cumprod_prev[t].view(-1, 1, 1, 1)

            posterior_mean = (torch.sqrt(alpha_bar_prev) * beta_t / (
                        1. - self.sde.alphas_cumprod[t].view(-1, 1, 1, 1))) * pred_x0 + \
                             (torch.sqrt(self.sde.alphas[t].view(-1, 1, 1, 1)) * (1. - alpha_bar_prev) / (
                                         1. - self.sde.alphas_cumprod[t].view(-1, 1, 1, 1))) * x

            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1)
                noise = torch.randn_like(x)
                x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x_prev = posterior_mean

            x = x_prev

        return x.clamp(-1, 1)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: center = (int(w / 2), int(h / 2))
    if radius is None: radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return torch.from_numpy(mask.astype(float))


# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================


def train(cfg, model, sde, train_loader, device, model_name, ckpt_mgr, results_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.params.lr)
    print(f"--- Starting Training for {model_name} ---")
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}"):
            optimizer.zero_grad();
            x0 = images.to(device)
            t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0);
            sqrt_alpha_bar_t = sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
            predicted_noise = model(xt, t.float());
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward();
            optimizer.step()
        if epoch % cfg.training.log_every_epoch == 0:
            visualize_training_epoch(model, cfg, sde, device, results_dir, model_name, epoch)
    print(f"--- Finished Training for {model_name} ---");
    ckpt_mgr.save(model, model_name)


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

    cfg = Box({
        "experiment": {
            "name": args.exp_name,
            "run": args.run_id,
        },
        "dataset": {
            "image_size": 32, "channels": 3,
            "model_A_digit": [6], "model_A_color": (1.0, 0.0, 0.0),  # Red 6
            "model_B_digit": [2], "model_B_color": (0.0, 0.8, 0.0),  # Green 2
        },
        "training":
            {"do_train": True,
             "epochs": 1 if args.sanity else 100,
            "log_every_epoch": 1 if args.sanity else 20,
            "sanity_num_examples": 8,
            "batch_size": 4 if args.sanity else 128
             },
        "optimizer": {"params": {"lr": 2e-4}},
        "sampling": {"batch_size": 4}
    })
    # --- Save configuration to YAML ---
    save_config_to_yaml(cfg, log_dir)


    sde = VPSDE(device=device)

    # --- Datasets and Training ---
    dataset_A = ColoredMNIST(cfg.dataset.image_size, target_digits=cfg.dataset.model_A_digit,
                             color_override=cfg.dataset.model_A_color)
    dataset_B = ColoredMNIST(cfg.dataset.image_size, target_digits=cfg.dataset.model_B_digit,
                             color_override=cfg.dataset.model_B_color)

    if args.sanity:
        dataset_A = tiny_subset(dataset_A, cfg.training.sanity_num_examples)
        dataset_B = tiny_subset(dataset_B, cfg.training.sanity_num_examples)

    loader_A = DataLoader(dataset_A, batch_size=cfg.training.batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=cfg.training.batch_size, shuffle=True)

    model_A = ScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_B = ScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_A_name, model_B_name = f"model_A_{cfg.experiment.name}", f"model_B_{cfg.experiment.name}"

    if cfg.training.do_train:
        train(cfg, model_A, sde, loader_A, device, model_A_name, ckpt_mgr, results_dir)
        train(cfg, model_B, sde, loader_B, device, model_B_name, ckpt_mgr, results_dir)

    # --- Inference ---
    print("\n--- 🎬 Starting Layout-Aware Inference ---")
    model_A = ckpt_mgr.load(model_A, model_A_name, device)
    model_B = ckpt_mgr.load(model_B, model_B_name, device)

    # --- Create Masks for Layout Control ---
    H, W = cfg.dataset.image_size, cfg.dataset.image_size
    # Mask for Green 2 (background)
    mask_B = create_circular_mask(H, W, center=(W // 2, H // 2), radius=H // 3)
    # Mask for Red 6 (foreground, on top)
    mask_A = create_circular_mask(H, W, center=(W // 3, H // 3), radius=H // 4)

    # The order matters: last model/mask is on top.
    # Model B (green 2) is the background, Model A (red 6) is the foreground.
    models_for_layout = [model_B, model_A]
    masks_for_layout = [mask_B, mask_A]

    # --- Instantiate and Run the Sampler ---
    layout_sampler = LayoutDiff(sde)
    layout_samples = layout_sampler.sample(
        models=models_for_layout,
        masks=masks_for_layout,
        shape=(cfg.sampling.batch_size, cfg.dataset.channels, H, W),
        device=device
    )

    # --- Visualize the Results ---
    def norm(s):
        return (s.clamp(-1, 1) + 1) / 2


    # Also visualize the masks themselves to understand the layout
    mask_viz = torch.stack([mask_B, mask_A]).unsqueeze(1).repeat(1, 3, 1, 1).float().to(device)


    path_comp = ckpt_mgr.get_path('results') / "composition"
    path_comp.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        torch.cat([norm(layout_samples), mask_viz]),
        path_comp / "composition_final.png",
        nrow=cfg.sampling.batch_size
    )
    print(f"Saved layout composition to {path_comp / 'composition_final.png'}")
    print("Top row: Generated images. Bottom row: Masks used (Green=Model B, Red=Model A).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose two diffusion models.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--run_id", type=str, required=True, help="A unique ID for the current run.")
    parser.add_argument("--project_name", type=str, default="mini-composable-diffusion-model",
                        help="Name of the project directory.")
    parser.add_argument("--skip_train", action='store_true',
                        help="Skip training and load models directly from checkpoints.")

    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")
    args = parser.parse_args()
    main(args)