# viz.py
import torch
import imageio
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def save_grid(tensor, path, nrow=8):
    """Saves a grid of images after de-normalizing."""
    tensor = (tensor + 1) / 2.0  # De-normalize from [-1, 1] to [0, 1]
    save_image(tensor, path, nrow=nrow)

def save_gif(frames, path, duration=150):
    """Saves a list of image tensors as a GIF."""
    imgs = [((frame.cpu().permute(1, 2, 0) + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8).numpy() for frame in frames]
    imageio.mimsave(path, imgs, duration=duration, loop=0)

def scatter2d(xy, path, title, labels=None):
    """Creates and saves a 2D scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(xy[:, 0], xy[:, 1], alpha=0.5, c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_loss(losses, path, title="Training Loss"):
    """Plots and saves the training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(path)
    plt.close()