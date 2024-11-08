#!/bin/bash
#SBATCH --job-name=install_pytorch
#SBATCH --output=install_pytorch_output.log
#SBATCH --error=install_pytorch_error.log
#SBATCH --time=00:30:00  # Adjust if you need more time
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Load required modules
module load cuDNN/8.8.0.121-CUDA-12.0.0
module load miniconda

# Activate the desired conda environment
conda deactivate
conda activate gaze_processing

# Install PyTorch and related packages
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

