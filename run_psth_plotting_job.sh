#!/bin/bash
#SBATCH --job-name=plot_psth_sacc
#SBATCH --partition=psych_day
#SBATCH --time=03:00:00                #6 hour time limit
#SBATCH --cpus-per-task=8            # CPUs
#SBATCH --mem=50G                     # total memory
#SBATCH --output=sacc_psth_plot.out  # Output file
#SBATCH --error=sacc_psth_plot.err   # Error file

# Load the necessary module and activate the conda environment
module load miniconda
conda deactivate
conda activate gaze_processing

# Run your script
python analyze_neural_spiking_response_to_behavior.py
