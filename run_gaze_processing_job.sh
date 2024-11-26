#!/bin/bash
#SBATCH --job-name=top_shuffled_crosscorr
#SBATCH --partition=psych_day
#SBATCH --time=10:00:00                # 1 hour time limit
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --mem=150G                     # total memory
#SBATCH --output=shuffled_crosscorr.out  # Output file
#SBATCH --error=shuffled_crosscorr.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python analyze_gaze_signals.py

