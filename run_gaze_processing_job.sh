#!/bin/bash
#SBATCH --job-name=top_shuffled_crosscorr
#SBATCH --partition=psych_day
#SBATCH --time=6:00:00                # 1 hour time limit
#SBATCH --cpus-per-task=4            # 80 CPUs
#SBATCH --mem-per-cpu=8G                     # 10GB of memory per CPU
#SBATCH --output=shuffled_crosscorr.out  # Output file
#SBATCH --error=shuffled_crosscorr.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python analyze_gaze_signals.py

