# shapes/dataset_grayscale.py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from PIL import Image, ImageDraw

class ShapesGrayscaleDataset(Dataset):
    """
    Generates grayscale images of simple shapes.
    """
    def __init__(self, size=10000, img_size=64):
        self.size = size
        self.img_size = img_size
        self.shapes = ["circle", "square", "triangle"]
        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}

        self.transform = Compose([
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

    def __len__(self):
        return self.size

    def _draw_shape(self, shape, draw):
        margin = self.img_size // 4
        top_left = (margin, margin)
        bottom_right = (self.img_size - margin, self.img_size - margin)
        if shape == "circle":
            draw.ellipse([top_left, bottom_right], fill="white")
        elif shape == "square":
            draw.rectangle([top_left, bottom_right], fill="white")
        elif shape == "triangle":
            p1 = (self.img_size // 2, margin)
            p2 = (margin, self.img_size - margin)
            p3 = (self.img_size - margin, self.img_size - margin)
            draw.polygon([p1, p2, p3], fill="white")

    def __getitem__(self, idx):
        shape_name = self.shapes[idx % len(self.shapes)]
        shape_label = torch.tensor(self.shape_to_idx[shape_name])

        # Create a grayscale ('L') image
        image = Image.new("L", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(image)
        self._draw_shape(shape_name, draw)

        return self.transform(image), shape_label
