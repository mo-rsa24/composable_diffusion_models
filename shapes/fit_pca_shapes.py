# fit_pca_shapes.py
import numpy as np
import os
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import joblib

# Assuming the new shapes/dataset.py is in the correct path
from shapes.dataset_ import ShapesDataset

print("--- Fitting PCA Model on Shapes Dataset ---")

# --- Configuration ---
N_COMPONENTS = 2
CHECKPOINT_DIR = "checkpoints_shapes_2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
PCA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pca_shapes.joblib")

# --- Load full dataset ---
full_dataset = ShapesDataset(size=10000)
# Use a large batch size to load all data at once
full_dataloader = DataLoader(full_dataset, batch_size=len(full_dataset))
all_images, _, _ = next(iter(full_dataloader))
# Flatten images for PCA: (N, C, H, W) -> (N, C*H*W)
images_flat = all_images.view(all_images.size(0), -1).numpy()

# --- Fit and Save PCA Model ---
pca = PCA(n_components=N_COMPONENTS)
pca.fit(images_flat)

# Save the entire PCA model using joblib
joblib.dump(pca, PCA_MODEL_PATH)

print(f"PCA model saved to {PCA_MODEL_PATH}")
print("You can now train the expert latent diffusion models.")
