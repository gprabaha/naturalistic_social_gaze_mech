#!/bin/bash
#SBATCH --job-name=mutual_behav
#SBATCH --partition=psych_day
#SBATCH --time=05:00:00                #6 hour time limit
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --mem=100G                     # total memory
#SBATCH --output=mutual_behav.out  # Output file
#SBATCH --error=mutual_behav.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python detect_mutual_behav_density.py
