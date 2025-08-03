# viz.py
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

def save_grid(tensor, path, nrow=8):
    """Saves a grid of images after de-normalizing."""
    tensor = (tensor + 1) / 2.0  # De-normalize from [-1, 1] to [0, 1]
    save_image(tensor, path, nrow=nrow)

def save_grid_(tensor, path, in_channels: int = 3):
    # If grayscale, add a channel dimension for save_image
    if in_channels == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    # Normalize to [0, 1] for saving
    save_image(tensor.clamp(-1, 1), path, normalize=True, value_range=(-1, 1))
    print(f"Validation grid saved to {path}")

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


def scatter2d_labeled(xy, labels, path, title):
    """
    Creates and saves a 2D scatter plot with a legend for each class.
    """
    plt.figure(figsize=(12, 10))
    # Use a colormap that has distinct colors for up to 10 classes
    cmap = plt.get_cmap('tab10')

    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        # Find the indices for each class
        idx = np.where(labels == label)
        # Plot the points for the current class
        plt.scatter(xy[idx, 0], xy[idx, 1], color=cmap(i), label=f'Digit {label}', alpha=0.6)

    plt.title(title, fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
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