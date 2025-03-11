#!/bin/bash
#SBATCH --job-name=mutual_behav_density
#SBATCH --output=mutual_behav_density.out
#SBATCH --error=mutual_behav_density.err
#SBATCH --partition=psych_day     # Change to psych_week manually if needed
#SBATCH --cpus-per-task=1        # Change manually if needed
#SBATCH --mem-per-cpu=100G         # Change manually if needed
#SBATCH --time=2:00:00           # Change manually if needed

echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "Time limit: $SLURM_JOB_TIMELIMIT"

module load miniconda
conda deactivate
conda activate gaze_processing

python detect_mutual_behav_density.py
