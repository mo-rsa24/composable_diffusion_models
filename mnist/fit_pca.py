# fit_pca.py
import numpy as np
import os
from sklearn.decomposition import PCA
from dataset import get_mnist_dataloader

print("--- Fitting Universal PCA Model on All MNIST Digits ---")

# --- Configuration ---
N_COMPONENTS = 2
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
PCA_PATH_MEAN = os.path.join(CHECKPOINT_DIR, "pca_mean.npy")
PCA_PATH_COMPONENTS = os.path.join(CHECKPOINT_DIR, "pca_components.npy")

# --- Load full dataset ---
full_dataloader = get_mnist_dataloader(batch_size=60000, shuffle=False, classes=None)
all_images, _ = next(iter(full_dataloader))
images_flat = all_images.view(all_images.size(0), -1).numpy()

# --- Fit and Save PCA Model ---
pca = PCA(n_components=N_COMPONENTS)
pca.fit(images_flat)

np.save(PCA_PATH_MEAN, pca.mean_)
np.save(PCA_PATH_COMPONENTS, pca.components_)

print(f"Universal PCA model saved to {CHECKPOINT_DIR}/")
print("You can now train the expert models.")