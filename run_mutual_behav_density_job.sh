#!/bin/bash
#SBATCH --job-name=5-25_mutual_behav
#SBATCH --partition=psych_day
#SBATCH --time=02:00:00                #2 hour time limit
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --mem=100G                     # total memory
#SBATCH --output=mutual_behav_5-25.out  # Output file
#SBATCH --error=mutual_behav_5-25.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python detect_mutual_behav_density.py
