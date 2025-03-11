#!/bin/bash
#SBATCH --job-name=mutual_behav_density
#SBATCH --output=mutual_behav_density.out
#SBATCH --error=mutual_behav_density.err

# Default settings
PARTITION="psych_day"
CPUS=16
MEMORY=16G
TIME="12:00:00"  # Default time for psych_day

# Check for external input
if [[ $1 == "week" ]]; then
    PARTITION="psych_week"
    CPUS=1
    MEMORY=450G
    TIME="3-00:00:00"  # 3 days
fi

#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem-per-cpu=$MEMORY
#SBATCH --time=$TIME

echo "Running on partition: $PARTITION"
echo "Using $CPUS CPUs with $MEMORY per CPU"
echo "Time limit: $TIME"

module load miniconda
conda deactivate
conda activate gaze_processing

python detect_mutual_behav_density.py
