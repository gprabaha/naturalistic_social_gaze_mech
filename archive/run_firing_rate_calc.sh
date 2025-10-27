#!/bin/bash
#SBATCH --job-name=firing_rate_calc
#SBATCH --output=firing_rate_calc.out
#SBATCH --error=firing_rate_calc.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --partition=psych_day
#SBATCH --time=2:00:00

module load miniconda
conda deactivate
conda activate gaze_processing

python modeling/preprocess_dataframes.py

