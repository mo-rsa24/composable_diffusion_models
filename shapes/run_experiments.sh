#!/bin/bash

# ===================================================================================
#  Composable Diffusion Shapes Experiment Runner
# ===================================================================================
# This script automates the training and visualization pipelines for both the
# latent space analysis and the final image model training.
#
# Make sure you have created the necessary directories:
# mkdir -p checkpoints_shapes outputs_shapes
#
# Usage:
#   - To run the entire latent space experiment: ./run_experiments.sh latent
#   - To run the entire image model training:   ./run_experiments.sh image
#   - To run everything sequentially:             ./run_experiments.sh all
# ===================================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Experiment Selection ---
EXPERIMENT_MODE=$1

# --- Pipeline 1: Latent Space Visualization ---
run_latent_space_experiment() {
  echo "====================================================="
  echo "  STARTING: Latent Space Visualization Experiment"
  echo "====================================================="

  # Step 1: Fit the PCA model on the entire dataset to create the latent space.
  echo -e "\n[Step 1/4] Fitting PCA model..."
  python3 shapes/fit_pca_shapes.py

  # Step 2: Train the two MLP expert models in the 2D latent space.
  echo -e "\n[Step 2/4] Training latent expert models..."
  # Train the shape expert (learns to distinguish circles vs. squares)
  python3 shapes/train_latent_shapes.py \
    --training_mode shape \
    --model_path checkpoints_shapes/shape_expert.pth \
    --epochs 300

  # Train the color expert (learns to distinguish red vs. green)
  python3 shapes/train_latent_shapes.py \
    --training_mode color \
    --model_path checkpoints_shapes/color_expert.pth \
    --epochs 300

  # Step 3: Visualize the forward diffusion process in the latent space.
  # This helps verify that the PCA projection was successful.
  echo -e "\n[Step 3/4] Visualizing forward diffusion in latent space..."
  python3 shapes/visualize_forward_shapes.py

  # Step 4: Visualize the compositional reverse diffusion process.
  # This uses the two trained MLP models to recreate the paper's visualization.
  echo -e "\n[Step 4/4] Visualizing compositional reverse diffusion (Itô Method)..."
  python3 shapes/visualize_composition_latent_ito.py

  # This uses the two trained MLP models to recreate the paper's visualization.
  echo -e "\n[Step 4/4] Visualizing compositional reverse diffusion (DDIM Method)..."
  python3 shapes/visualize_composition_shapes.py

  echo -e "\n====================================================="
  echo "  COMPLETED: Latent Space Visualization Experiment"
  echo "  Check the 'outputs_shapes/' directory for results."
  echo "====================================================="
}

# --- Pipeline 2: Training Diffusion Models on Shape and Color ---
run_image_model_training() {
  echo "====================================================="
  echo "  STARTING: Image Model Training Experiment"
  echo "====================================================="

  # Step 1: Train the U-Net shape expert model.
  # This model learns to generate grayscale shapes.
  echo -e "\n[Step 1/3] Training U-Net shape model..."
  python3 shapes/train_image.py \
    --training_mode shape \
    --model_path checkpoints_shapes/unet_shape_expert.pth \
    --epochs 400 \

  # Step 2: Train the U-Net color expert model.
  # This model learns to generate color blobs.
  echo -e "\n[Step 2/3] Training U-Net color model..."
  python3 shapes/train_image.py \
    --training_mode color \
    --model_path checkpoints_shapes/unet_color_expert.pth \
    --epochs 400 \

  # Step 3: Compose U-Net color and shape expert model.
  # This model learns to generate color blobs.
  echo -e "\n[Step 3/3] Composing U-Net shape and color models with DDIM sampler..."
  python3 shapes/compose_images_ddim.py \
    --shape_model_path checkpoints_shapes/unet_shape_expert.pth \
    --color_model_path checkpoints_shapes/unet_color_expert.pth


    # Step 3: Compose U-Net color and shape expert model.
  # This model learns to generate color blobs.
  echo -e "\n[Step 3/3] Composing U-Net shape and color models with Ito Sampler..."
  python3 shapes/compose_images_ito.py \
    --shape_model_path checkpoints_shapes/unet_shape_expert.pth \
    --color_model_path checkpoints_shapes/unet_color_expert.pth



  echo -e "\n====================================================="
  echo "  COMPLETED: Image Model Training Experiment"
  echo "  Trained models are saved in 'checkpoints_shapes/'."
  echo "  Validation samples are in 'outputs_shapes/'."
  echo "  You can now use these models with a composition script."
  echo "====================================================="
}


# --- Main Execution Logic ---
if [ "$EXPERIMENT_MODE" == "latent" ]; then
  run_latent_space_experiment
elif [ "$EXPERIMENT_MODE" == "image" ]; then
  run_image_model_training
elif [ "$EXPERIMENT_MODE" == "all" ]; then
  run_latent_space_experiment
  run_image_model_training
else
  echo "Invalid argument. Please specify 'latent', 'image', or 'all'."
  echo "Usage: ./run_experiments.sh [latent|image|all]"
  exit 1
fi
