# dataset.py (Modified for Shape/Color Composition)
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFilter


class ShapesDataset(Dataset):
    """Generates images of simple shapes with specified colors on the fly."""

    def __init__(self, size=5000, img_size=64, mode='rgb'):
        self.size = size
        self.img_size = img_size
        self.mode = mode  # 'rgb', 'shape', or 'color'

        self.shapes = ["circle", "square", "triangle"]
        self.colors = ["red", "green", "blue"]

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]

        self.transform_rgb = Compose([
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

        self.transform_shape = Compose([
            Grayscale(num_output_channels=1),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
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
            p1 = (self.img_size // 2, margin)
            p2 = (margin, self.img_size - margin)
            p3 = (self.img_size - margin, self.img_size - margin)
            draw.polygon([p1, p2, p3], fill=color_name)

    def __getitem__(self, idx):
        shape_name, color_name = self.all_combinations[idx % len(self.all_combinations)]

        # Create the base RGB image
        image = Image.new("RGB", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(image)
        self._draw_shape(shape_name, color_name, draw)

        if self.mode == 'shape':
            # Return grayscale version for shape-only training
            return self.transform_shape(image)

        elif self.mode == 'color':
            # For color training, we destroy shape info by blurring heavily
            # This creates a "color blob"
            blob_image = image.filter(ImageFilter.GaussianBlur(radius=self.img_size / 4))
            return self.transform_rgb(blob_image)

        else:  # self.mode == 'rgb'
            # Return the original colored shape
            return self.transform_rgb(image)