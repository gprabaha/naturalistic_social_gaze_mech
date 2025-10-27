#!/bin/bash
#SBATCH --job-name=plot_crosscorr_fix_vec
#SBATCH --partition=psych_day
#SBATCH --time=2:00:00                #6 hour time limit
#SBATCH --cpus-per-task=16            # CPUs
#SBATCH --mem=100G                     # total memory
#SBATCH --output=plot_crosscorr_fix_vec.out  # Output file
#SBATCH --error=plot_crosscorr_fix_vec.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python compute_inter_agent_fixation_timeline_crosscorrelations.py

