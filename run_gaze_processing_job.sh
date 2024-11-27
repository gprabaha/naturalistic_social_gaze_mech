#!/bin/bash
#SBATCH --job-name=crosscorr_reg_and_shuffled
#SBATCH --partition=psych_day
#SBATCH --time=10:00:00                # 1 hour time limit
#SBATCH --cpus-per-task=18            # CPUs
#SBATCH --mem=180G                     # total memory
#SBATCH --output=crosscorr_all.out  # Output file
#SBATCH --error=crosscorr_all.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python analyze_gaze_signals.py

