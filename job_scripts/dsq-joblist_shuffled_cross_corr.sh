#!/bin/bash
#SBATCH --output ./job_scripts/
#SBATCH --array 0-416
#SBATCH --job-name dsq-joblist_shuffled_cross_corr
#SBATCH --partition psych_day --cpus-per-task 18 --mem-per-cpu 10G -t 00:30:00

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/naturalistic_social_gaze_mech/job_scripts/joblist_shuffled_cross_corr.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/naturalistic_social_gaze_mech/job_scripts

