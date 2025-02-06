#!/bin/bash
#SBATCH --output ./job_scripts_slds/
#SBATCH --array 0
#SBATCH --job-name dsq-slds_joblist
#SBATCH --partition psych_day --cpus-per-task 3 --mem-per-cpu 8G -t 01:00:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/naturalistic_social_gaze_mech/job_scripts_slds/slds_joblist.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/naturalistic_social_gaze_mech/job_scripts_slds

