#!/bin/bash
# Install NVIDIA GPU support for ML
# Run with: sudo bash install-gpu.sh

set -e

echo "=== Installing NVIDIA CUDA Toolkit ==="
apt install -y nvidia-cuda-toolkit

echo "=== Verifying GPU ==="
nvidia-smi

echo "=== Done! ==="
echo "Install PyTorch with CUDA:"
echo "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
echo ""
echo "Install TensorFlow with GPU:"
echo "  uv pip install tensorflow"