#!/bin/bash
#SBATCH --job-name=crosscorr_fix_vec
#SBATCH --partition=psych_day
#SBATCH --time=5:00:00                #6 hour time limit
#SBATCH --cpus-per-task=4            # CPUs
#SBATCH --mem=478G                     # total memory
#SBATCH --output=crosscorr_fix_vec.out  # Output file
#SBATCH --error=crosscorr_fix_vec.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python compute_inter_agent_fixation_timeline_crosscorrelations.py

