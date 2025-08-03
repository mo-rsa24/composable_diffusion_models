#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Full Latent Composition Experiment ---"

# --- Step 1: Create the Universal PCA model (run only once) ---
echo ">>> Step 1: Fitting the Universal PCA Model..."
python3 mnist/fit_pca.py

# --- Step 2: Train the first expert latent model (digits 0-4) ---
echo ">>> Step 2: Training expert model for digits 0-4..."
python3 mnist/train_latent_2d.py \
    --classes 0 1 2 3 4 \
    --model_path "checkpoints/latent_model_0_4.pth" \
    --epochs 150

# --- Step 3: Train the second expert latent model (digits 5-9) ---
echo ">>> Step 3: Training expert model for digits 5-9..."
python3 mnist/train_latent_2d.py \
    --classes 5 6 7 8 9 \
    --model_path "checkpoints/latent_model_5_9.pth" \
    --epochs 150

# --- Step 4: Run the visualization script ---
# No changes are needed for visualize_composition_latent.py
echo ">>> Step 4: Visualizing the combined reverse diffusion process..."
python3 mnist/visualize_composition_latent.py

echo "--- Latent Composition Experiment Finished Successfully! ---"
echo "Check the 'outputs/composition/' directory for the final visualization."