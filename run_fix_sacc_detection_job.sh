#!/bin/bash
#SBATCH --job-name=fix_sacc_plot
#SBATCH --partition=psych_day
#SBATCH --time=01:00:00                #6 hour time limit
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --mem=50G                     # total memory
#SBATCH --output=fix_sacc_plot.out  # Output file
#SBATCH --error=fix_sacc_plot.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python detect_eye_mvm_behav_from_gaze_data.py
