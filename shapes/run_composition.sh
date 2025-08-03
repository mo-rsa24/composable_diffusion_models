#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Composition Experiment ---"

# --- Define common variables ---
CHECKPOINT_DIR="checkpoints"
OUTPUT_DIR="outputs/composition"
EPOCHS=50 # Use fewer epochs for a quick test if needed

# --- Step 1: Train the first expert model (digits 0-4) ---
echo ">>> Training expert model for digits 0-4..."
python3  mnist/train_image.py \
    --exp_name "0to4" \
    --classes 0 1 2 3 4 \
    --model_path "$CHECKPOINT_DIR/model_0_4.pth" \
    --epochs $EPOCHS

# --- Step 2: Train the second expert model (digits 5-9) ---
echo ">>> Training expert model for digits 5-9..."
python3 mnist/train_image.py \
    --exp_name "5to9" \
    --classes 5 6 7 8 9 \
    --model_path "$CHECKPOINT_DIR/model_5_9.pth" \
    --epochs $EPOCHS

# --- Step 3: Run the composition script ---
echo ">>> Composing the two expert models..."
python3 mnist/compose_scores.py \
    --model1_path "$CHECKPOINT_DIR/model_0_4.pth" \
    --model2_path "$CHECKPOINT_DIR/model_5_9.pth" \
    --output_file "$OUTPUT_DIR/composed_samples_w1_w1.png" \
    --w1 1.0 \
    --w2 1.0

echo "--- Composition Experiment Finished Successfully ---"
echo "Check the '$OUTPUT_DIR' directory for the generated image."