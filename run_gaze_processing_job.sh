#!/bin/bash
#SBATCH --job-name=fix_sacc_recompute
#SBATCH --partition=psych_day
#SBATCH --time=10:00:00                #6 hour time limit
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --mem=50G                     # total memory
#SBATCH --output=fix_sacc.out  # Output file
#SBATCH --error=fix_sacc.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your gcript
python analyze_gaze_signals.py
