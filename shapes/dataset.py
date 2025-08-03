import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
from PIL import Image, ImageDraw



class ShapesDataset(Dataset):
    """Generates images of simple shapes with specified colors on the fly."""

    def __init__(self, Config, size=1000, img_size=64, shapes=None, colors=None, holdout=None):
        self.size = size
        self.img_size = img_size
        self.shapes = shapes if shapes is not None else Config.SHAPES
        self.colors = colors if colors is not None else Config.COLORS
        self.holdout = holdout

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]
        if self.holdout:
            self.all_combinations.remove(self.holdout)

        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

        self.transform = Compose([
            Resize(img_size), CenterCrop(img_size), ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

    def __len__(self):
        return self.size

    def _draw_shape(self, shape, color_name, draw):
        img_size = self.img_size
        margin = img_size // 4
        top_left, bottom_right = (margin, margin), (img_size - margin, img_size - margin)
        if shape == "circle":
            draw.ellipse([top_left, bottom_right], fill=color_name)
        elif shape == "square":
            draw.rectangle([top_left, bottom_right], fill=color_name)
        elif shape == "triangle":
            p1, p2, p3 = (img_size // 2, margin), (margin, img_size - margin), (img_size - margin, img_size - margin)
            draw.polygon([p1, p2, p3], fill=color_name)

    def __getitem__(self, idx):
        shape_name, color_name = self.all_combinations[idx % len(self.all_combinations)]
        image = Image.new("RGB", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(image)
        self._draw_shape(shape_name, color_name, draw)
        shape_idx = self.shape_to_idx[shape_name]
        color_idx = self.color_to_idx[color_name]
        return self.transform(image), torch.tensor(shape_idx), torch.tensor(color_idx)


def sample_2d_data(bs, up=True, device='cpu'):
    """Generates 2D point cloud data for the toy experiment."""
    centers = torch.tensor([[0, 2.5], [0, -2.5]], device=device)
    center = centers[0] if up else centers[1]

    # Generate points in a cluster
    data = torch.randn(bs, 2, device=device) * 0.4
    data += center
    return data