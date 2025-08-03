# shapes/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from PIL import Image, ImageDraw


class ShapesDataset(Dataset):
    """
    Generates images of simple shapes with specified colors and returns their labels.
    """

    def __init__(self, size=10000, img_size=64):
        self.size = size
        self.img_size = img_size

        self.shapes = ["circle", "square", "triangle"]
        self.colors = ["red", "green", "blue"]

        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]

        self.transform = Compose([
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

    def __len__(self):
        return self.size

    def _draw_shape(self, shape, color_name, draw):
        margin = self.img_size // 4
        top_left = (margin, margin)
        bottom_right = (self.img_size - margin, self.img_size - margin)
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

        shape_label = torch.tensor(self.shape_to_idx[shape_name])
        color_label = torch.tensor(self.color_to_idx[color_name])

        image = Image.new("RGB", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(image)
        self._draw_shape(shape_name, color_name, draw)

        return self.transform(image), shape_label, color_label
