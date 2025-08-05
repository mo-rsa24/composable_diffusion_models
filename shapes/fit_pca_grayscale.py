# shapes/fit_pca_grayscale.py
import numpy as np
import os
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import joblib

# Use the new grayscale dataset
from shapes.dataset_grayscale import ShapesGrayscaleDataset

print("--- Fitting PCA Model on Grayscale Shapes Dataset ---")

# --- Configuration ---
N_COMPONENTS = 2
CHECKPOINT_DIR = "checkpoints_shapes_grayscale"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
PCA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pca_grayscale.joblib")

# --- Load full dataset ---
full_dataset = ShapesGrayscaleDataset(size=10000)
full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
all_images, _ = next(iter(full_dataloader))
# Flatten images for PCA: (N, C, H, W) -> (N, C*H*W)
# For grayscale, C=1, so we have 64*64=4096 features
images_flat = all_images.view(all_images.size(0), -1).numpy()

# --- Fit and Save PCA Model ---
pca = PCA(n_components=N_COMPONENTS)
pca.fit(images_flat)

joblib.dump(pca, PCA_MODEL_PATH)
print(f"Grayscale PCA model saved to {PCA_MODEL_PATH}")
