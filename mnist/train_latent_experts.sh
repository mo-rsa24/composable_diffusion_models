#!/bin/bash
set -e

echo "--- Training Latent Expert for Digits 0-4 ---"
python3 train_latent_2d.py \
    --classes 0 1 2 3 4 \
    --model_path "checkpoints/latent_model_0_4.pth" \
    --epochs 100

echo "--- Training Latent Expert for Digits 5-9 ---"
python3 train_latent_2d.py \
    --classes 5 6 7 8 9 \
    --model_path "checkpoints/latent_model_5_9.pth" \
    --epochs 100

echo "--- Latent Expert Training Complete ---"