#!/bin/bash
#SBATCH --output ./job_scripts/
#SBATCH --array 0
#SBATCH --job-name dsq-joblist
#SBATCH --partition psych_day --cpus-per-task 4 --mem-per-cpu 8000 -t 4:00:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/naturalistic_social_gaze_mech/job_scripts/joblist.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/naturalistic_social_gaze_mech/job_scripts
