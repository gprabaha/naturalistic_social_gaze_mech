#!/bin/bash
#SBATCH --job-name=gaze_processing
#SBATCH --partition=psych_day
#SBATCH --time=1:00:00                # 1 hour time limit
#SBATCH --cpus-per-task=48            # 80 CPUs
#SBATCH --mem=64G                     # 10GB of memory per CPU
#SBATCH --output=gaze_processing.out  # Output file
#SBATCH --error=gaze_processing.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python analyze_gaze_signals.py

