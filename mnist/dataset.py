from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_mnist_dataloader(batch_size=128, shuffle=True, classes=None):
    """
    Returns a DataLoader for the MNIST dataset.
    Args:
        classes (list, optional): A list of classes to include (e.g., [0, 1, 2, 3, 4]).
                                  If None, all classes are used.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixels to [-1, 1]
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    if classes is not None:
        idx = np.zeros_like(dataset.targets, dtype=bool)
        for target_class in classes:
            idx = idx | (dataset.targets == target_class)
        dataset = Subset(dataset, np.where(idx)[0])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dataloader


def sample_data(dataloader):
    """Yields batches of MNIST images."""
    while True:
        for batch in dataloader:
            yield batch[0]  # Return only images, not labels