#!/bin/bash
#SBATCH --job-name=fit_slds
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=2
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=fit_slds_out.txt
#SBATCH --error=fit_slds_err.txt

# Load necessary modules
module load miniconda

# Activate conda environment
conda deactivate
conda activate gaze_processing

# Run the Python script
python fit_slds_to_fix_timeline.py
