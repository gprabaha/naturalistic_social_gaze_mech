#!/bin/bash
#SBATCH --job-name=5-25_trans_psth
#SBATCH --partition=psych_day
#SBATCH --time=03:00:00                #6 hour time limit
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --mem=100G                     # total memory
#SBATCH --output=trans_psth_5-25.out  # Output file
#SBATCH --error=trans_psth_5-25.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python detect_behavioral_transitions_within_sessions.py
