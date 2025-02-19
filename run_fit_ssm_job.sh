#!/bin/bash
#SBATCH --job-name=fit_ssm
#SBATCH --output=fit_ssm.out
#SBATCH --error=fit_ssm.out
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=psych_day
#SBATCH --time=10:00:00

ml miniconda
conda deactivate
conda activate gaze_processing

python fit_ssm_models_to_simplified_behav_timeline.py
