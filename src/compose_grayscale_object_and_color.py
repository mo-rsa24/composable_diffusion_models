"""
Intent
Compose two separately trained diffusion models—one on shape only,
 one on color only—to generate colored digits that neither model saw during training.

Hypothesis
If the grayscale “shape” model has learned pure digit morphology and the “color” model has learned color textures,
then fusing their score estimates at sampling time will produce a digit in the desired color,
 even for digit–color pairs unseen during training.
"""
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from box import Box
import math
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from src.diffusion.samplers import SuperDiffSampler
from src.models.compose_grayscale_object_and_color import ColoredMNISTScoreModel, VPSDE
from src.utils.tools import save_config_to_yaml, is_cluster, tiny_subset, CheckpointManager
from src.utils.visualization import visualize_training_epoch

model_A_name, model_B_name = "Object", "Color"


# ==============================================================================
# 2. CUSTOM DATASETS
# ==============================================================================
class GrayscaleMNIST(Dataset):
    """MNIST dataset filtered by digit, but kept as 3-channel grayscale."""

    def __init__(self, image_size, target_digits):
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        self.indices = [i for i, (_, label) in enumerate(mnist) if label in target_digits]
        self.mnist_dataset = mnist

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[self.indices[idx]]
        image_tensor = self.transform(image)
        # Repeat the single channel to get a 3-channel grayscale image and normalize to [-1, 1]
        final_image = (image_tensor.repeat(3, 1, 1) * 2) - 1
        return final_image, label

class RandomlyColoredMNIST(Dataset):
    """
    MNIST dataset where each digit is given a random color,
    forcing the model to learn shape independent of color.
    """
    def __init__(self, image_size, target_digits):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        self.indices = [i for i, (_, label) in enumerate(mnist) if label in target_digits]
        self.mnist_dataset = mnist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[self.indices[idx]]
        image_tensor = self.transform(image) # This is a 1-channel tensor [1, H, W]

        # Generate a random color
        random_color = torch.rand(3, 1, 1) # [3, 1, 1] tensor for R, G, B

        # Apply the color to the digit (which is where image_tensor > 0)
        colored_image = image_tensor.repeat(3, 1, 1) * random_color

        # Normalize to [-1, 1]
        final_image = (colored_image * 2) - 1
        return final_image, label

class SimpleShapesDataset(Dataset):
    """A synthetic dataset of simple, solid-colored shapes."""

    def __init__(self, image_size, num_samples, shape_color):
        self.image_size = image_size
        self.num_samples = num_samples
        self.shape_color = torch.tensor(shape_color).float().view(3, 1, 1)  # [C, H, W]

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        # Create a black canvas
        image = torch.zeros(3, self.image_size, self.image_size)
        # Draw a square of random size and position
        square_size = np.random.randint(self.image_size // 4, self.image_size // 2)
        x_start = np.random.randint(0, self.image_size - square_size)
        y_start = np.random.randint(0, self.image_size - square_size)
        image[:, y_start:y_start + square_size, x_start:x_start + square_size] = self.shape_color
        # Normalize to [-1, 1]
        return (image * 2) - 1, 0  # Label is unused

def get_dataset(name, image_size, **kwargs):
    if name.lower() == 'grayscalemnist':
        return GrayscaleMNIST(image_size, **kwargs)
    elif name.lower() == 'simpleshapesdataset':
        return SimpleShapesDataset(image_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
# ==============================================================================
# 3. TRAINING AND SAMPLING (Mostly Unchanged)
# ==============================================================================
def train(cfg, model, sde, train_loader, device, model_name, ckpt_mgr, results_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.params.lr)
    print(f"--- Starting Training for {model_name} ---")
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}"):
            optimizer.zero_grad()
            x0 = images.to(device)
            t = torch.randint(0, sde.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            sqrt_alpha_bar_t = sde.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
            predicted_noise = model(xt, t.float())
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()
        if epoch % cfg.training.log_every_epoch == 0:
            visualize_training_epoch(model, cfg, sde, device, results_dir, model_name, epoch)


    print(f"--- Finished Training for {model_name} ---")
    ckpt_mgr.save(model, model_name)

# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================

def visualize_results(samples_a, samples_b, superdiff_samples, ckpt_mgr):
    """Saves a grid comparing the three sets of samples to the correct directory."""
    def norm(s): return (s.clamp(-1, 1) + 1) / 2

    # Save individual model results
    path_a = ckpt_mgr.get_path( 'results') / model_A_name
    path_a.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(norm(samples_a), path_a / "samples_final.png", nrow=samples_a.shape[0])
    print(f"Saved Model A samples to {path_a}")

    path_b = ckpt_mgr.get_path('results') / model_B_name
    path_b.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(norm(samples_b), path_b / "samples_final.png", nrow=samples_b.shape[0])
    print(f"Saved Model B samples to {path_b}")

    # Save composition results
    path_comp = ckpt_mgr.get_path( 'results') / "composition"
    path_comp.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(norm(superdiff_samples), path_comp / "composition_final.png", nrow=superdiff_samples.shape[0])
    print(f"Saved Composed samples to {path_comp}")

    # Save comparison grid
    comparison_path = ckpt_mgr.get_path('results') / "composition"
    comparison_path.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(torch.cat([
        norm(samples_a), norm(samples_b), norm(superdiff_samples)
    ]), comparison_path / "comparison_grid.png" , nrow=samples_a.shape[0])
    print(f"Saved final comparison grid to {comparison_path} (Top: Model A, Middle: Model B, Bottom: Composed)")

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

    sde = VPSDE(device=device)

    cfg = Box({
        "experiment": {
            "name": args.exp_name,
            "run": args.run_id,
        },
        "dataset": {
            "image_size": 32,
            "channels": 3,
            "dataset_a": args.dataset_a,
            "dataset_b": args.dataset_b,
            "content_digit": [7],
            "style_color": (1.0, 0.0, 0.0),  # Red
            "num_samples_color": 5000  # Define a fixed number for the color dataset
        },
        "training": {
            "do_train": not args.skip_train,
            "epochs": 1 if args.sanity else 100,
            "log_every_epoch": 1 if args.sanity else 20,
            "sanity_num_examples": 8,
            "batch_size": 4 if args.sanity else 128
        },
        "optimizer": {
            "params": {"lr": 2e-4}
        },
        "sampling": {
            "batch_size": 8
        }
    })
    # --- Save configuration to YAML ---
    save_config_to_yaml(cfg, log_dir)

    print(f"--- 🧪 RUNNING EXPERIMENT: {args.exp_name} | RUN ID: {args.run_id} ---")

    # --- Corrected Dataset Initialization ---

    print(f"Initializing dataset A: {cfg.dataset.dataset_a}")
    if cfg.dataset.dataset_a == 'GrayscaleMNIST':
        dataset_A = GrayscaleMNIST(
            image_size=cfg.dataset.image_size,
            target_digits=cfg.dataset.content_digit
        )
        num_style_samples = len(dataset_A)
    # --- ADD THIS NEW CONDITION ---
    elif cfg.dataset.dataset_a == 'RandomlyColoredMNIST':
        dataset_A = RandomlyColoredMNIST(
            image_size=cfg.dataset.image_size,
            target_digits=cfg.dataset.content_digit
        )
        num_style_samples = len(dataset_A)
    # -----------------------------
    elif cfg.dataset.dataset_a == 'SimpleShapesDataset':
        num_style_samples = cfg.dataset.num_samples_color
        dataset_A = SimpleShapesDataset(
            image_size=cfg.dataset.image_size,
            num_samples=num_style_samples,
            shape_color=cfg.dataset.style_color
        )
    else:
        raise ValueError(f"Unknown dataset for dataset_a: {cfg.dataset.dataset_a}")

    # Create Dataset B
    print(f"Initializing dataset B: {cfg.dataset.dataset_b}")
    if cfg.dataset.dataset_b == 'SimpleShapesDataset':
        dataset_B = SimpleShapesDataset(
            image_size=cfg.dataset.image_size,
            num_samples=num_style_samples,
            shape_color=cfg.dataset.style_color
        )
    elif cfg.dataset.dataset_b == 'GrayscaleMNIST':
        dataset_B = GrayscaleMNIST(
            image_size=cfg.dataset.image_size,
            target_digits=cfg.dataset.content_digit
        )
    else:
        raise ValueError(f"Unknown dataset for dataset_b: {cfg.dataset.dataset_b}")

    if args.sanity:
        dataset_A = tiny_subset(dataset_A, cfg.training.sanity_num_examples)
        dataset_B = tiny_subset(dataset_B, cfg.training.sanity_num_examples)

    loader_A = DataLoader(dataset_A, batch_size=cfg.training.batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=cfg.training.batch_size, shuffle=True)
    operation = 'AND'

    model_A = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)
    model_B = ColoredMNISTScoreModel(in_channels=cfg.dataset.channels).to(device)

    if cfg.training.do_train:
        print("\n--- 🏋️ Starting Training Phase ---")
        train(cfg, model_A, sde, loader_A, device, model_A_name, ckpt_mgr, results_dir)
        train(cfg, model_B, sde, loader_B, device, model_B_name, ckpt_mgr, results_dir)
    else:
        print("\n--- Skipping Training Phase ---")

    print("\n--- 🎬 Starting Inference Phase ---")
    model_A = ckpt_mgr.load(model_A, model_A_name, device)
    model_B = ckpt_mgr.load(model_B, model_B_name, device)

    sampler = SuperDiffSampler(sde)
    shape = (cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)

    print("Generating samples from Model A (Content)...")
    samples_A = sampler.sample_single_model(model_A, cfg.sampling.batch_size, shape, device)

    print("Generating samples from Model B (Style)...")
    samples_B = sampler.sample_single_model(model_B, cfg.sampling.batch_size, shape, device)

    print(f"Generating composed samples using SUPERDIFF '{operation}'...")
    composed_samples = sampler.sample(model_A, model_B, cfg.sampling.batch_size, shape, device, operation)

    visualize_results(samples_A[:4], samples_B[:4], composed_samples[:4], ckpt_mgr)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose two diffusion models.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--run_id", type=str, required=True, help="A unique ID for the current run.")
    parser.add_argument("--project_name", type=str, default="mini-composable-diffusion-model",
                        help="Name of the project directory.")
    parser.add_argument("--dataset_a", type=str, default="RandomlyColoredMNIST",  # Changed default
                        choices=["GrayscaleMNIST", "SimpleShapesDataset", "RandomlyColoredMNIST"],
                        help="Dataset for the first model (content).")
    parser.add_argument("--dataset_b", type=str, default="SimpleShapesDataset",
                        choices=["GrayscaleMNIST", "SimpleShapesDataset"], help="Dataset for the second model (style).")
    parser.add_argument("--skip_train", action='store_true',
                        help="Skip training and load models directly from checkpoints.")
    parser.add_argument("--sanity", action='store_true',
                        help="Run sanity checks to ensure that the model is running")

    args = parser.parse_args()
    main(args)