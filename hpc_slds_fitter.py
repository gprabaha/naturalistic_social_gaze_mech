import os
import subprocess
import time
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class HPCSLDSFitter:
    def __init__(self, params):
        self.params = params
        self.job_script_out_dir = './job_scripts/'
        self.python_script_path = 'run_single_slds_fitting_task.py'  # New Python script for SLDS fitting

    def generate_job_file(self, fixation_timeline_df, params_file_path):
        """
        Generates a job file for submitting array jobs, where each task processes one session-run group.
        """
        job_file_path = os.path.join(self.job_script_out_dir, 'slds_joblist.txt')
        os.makedirs(self.job_script_out_dir, exist_ok=True)

        env_name = 'gaze_otnal' if self.params['is_grace'] else 'gaze_processing'

        grouped = fixation_timeline_df.groupby(["session_name", "interaction_type", "run_number"])
        
        with open(job_file_path, 'w') as file:
            for (session, interaction_type, run), _ in grouped:
                task_key = f"{session},{interaction_type},{run}"
                command = (
                    f"module load miniconda; "
                    f"conda activate {env_name}; "
                    f"python {self.python_script_path} {task_key} {params_file_path}"
                )
                file.write(command + "\n")

        logger.info("Generated SLDS job file at %s", job_file_path)
        return job_file_path

    def submit_job_array(self, job_file_path):
        """
        Submits the generated job file as a job array to the cluster using dSQ.
        """
        try:
            job_script_path = os.path.join(self.job_script_out_dir, 'dsq-joblist_slds.sh')
            partition = 'day' if self.params['is_grace'] else 'psych_day'

            logger.info("Generating the dSQ job script")
            subprocess.run(
                f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} '
                f'-o {self.job_script_out_dir} --status-dir {self.job_script_out_dir} --partition {partition} '
                f'--cpus-per-task 6 --mem-per-cpu 4000 -t 01:00:00 --mail-type FAIL',
                shell=True, check=True, executable='/bin/bash'
            )

            logger.info("Successfully generated the dSQ job script.")

            if not os.path.isfile(job_script_path):
                logger.error("No job script found at %s", job_script_path)
                return

            logger.info("Submitting jobs using dSQ job script: %s", job_script_path)
            result = subprocess.run(
                f'sbatch --job-name=dsq_slds '
                f'--output={self.job_script_out_dir}/slds_%a.out '
                f'--error={self.job_script_out_dir}/slds_%a.err '
                f'{job_script_path}',
                shell=True, check=True, capture_output=True, text=True, executable='/bin/bash'
            )

            logger.info("Successfully submitted jobs using sbatch for script %s", job_script_path)
            job_id = result.stdout.strip().split()[-1]
            logger.info("Submitted job array with ID: %s", job_id)
            self.track_job_progress(job_id)
        except subprocess.CalledProcessError as e:
            logger.error("Error during job submission process: %s", e)
            raise

    def track_job_progress(self, job_id):
        """
        Tracks the progress of the submitted job array.
        """
        logger.info("Tracking progress of job array with ID: %s", job_id)
        start_time = time.time()
        check_interval = 30  # Check the job status every 30 seconds
        print_every_n_mins = 1
        print_interval = print_every_n_mins * 60  # Print job status every 1 minute
        last_print_time = start_time

        while True:
            result = subprocess.run(
                f'squeue --job {job_id} -h -o %T',
                shell=True, capture_output=True, text=True, executable='/bin/bash'
            )

            if result.returncode != 0:
                logger.error("Error checking job status for job ID %s: %s", job_id, result.stderr.strip())
                break

            job_statuses = result.stdout.strip().split()
            if not job_statuses:
                logger.info("Job array %s has completed.", job_id)
                break

            running_jobs = [status for status in ('PENDING', 'RUNNING', 'CONFIGURING') if status in job_statuses]
            current_time = time.time()

            if not running_jobs:
                logger.info("Job array %s has completed.", job_id)
                break
            elif current_time - last_print_time >= print_interval:
                logger.info("Job array %s is still running. Checking again in %d minutes...", job_id, print_every_n_mins)
                last_print_time = current_time

            time.sleep(check_interval)
