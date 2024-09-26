import os
import subprocess
import logging
import time

class HPCFixAndSaccadeDetector:
    def __init__(self, params):
        self.params = params
        self.job_script_out_dir = './job_scripts/'
        self.logger = logging.getLogger(__name__)
        self.python_script_path = 'run_single_fix_and_saccade_detection_task.py'  # Path to the new Python script


    def generate_job_file(self, tasks, params_file_path):
        """
        Generates a job file for submitting array jobs, with each task processing one run.
        """
        job_file_path = os.path.join(self.job_script_out_dir, 'joblist.txt')
        os.makedirs(self.job_script_out_dir, exist_ok=True)
        env_name = 'gaze_otnal' if self.params['is_grace'] else 'gaze_processing'
        with open(job_file_path, 'w') as file:
            for task in tasks:
                session, interaction_type, run, agent = task
                task_key = f"{session},{interaction_type},{run},{agent}"
                command = (
                    f"module load miniconda; "
                    f"conda activate {env_name}; "
                    f"python {self.python_script_path} {task_key} {params_file_path}"
                )
                file.write(command + "\n")
        self.logger.info(f"Generated job file at {job_file_path}")
        return job_file_path


    def submit_job_array(self, job_file_path):
        """
        Submits the generated job file as a job array to the cluster using dSQ.
        """
        try:
            job_script_path = os.path.join(self.job_script_out_dir, 'dsq-joblist_fixations.sh')
            partition = 'day' if self.params['is_grace'] else 'psych_day'
            # Generate the dSQ job script, which includes the job array command
            subprocess.run(
                f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} '
                f'-o {self.job_script_out_dir} --status-dir {self.job_script_out_dir} --partition {partition} '
                f'--cpus-per-task 4 --mem-per-cpu 8000 -t 4:00:00 --mail-type FAIL',
                shell=True, check=True, executable='/bin/bash'
            )
            self.logger.info("Successfully generated the dSQ job script.")
            # Check that the job script exists before submitting
            if not os.path.isfile(job_script_path):
                self.logger.error(f"No job script found at {job_script_path}.")
                return
            self.logger.info(f"Using dSQ job script: {job_script_path}")
            # Submit the job array with descriptive output and error filenames using array index %a
            result = subprocess.run(
                f'sbatch --job-name=dsq_fix_saccade '
                f'--output={self.job_script_out_dir}/fixation_saccade_%a.out '  # %A is the job ID, %a is the task index
                f'--error={self.job_script_out_dir}/fixation_saccade_%a.err '
                f'{job_script_path}',
                shell=True, check=True, capture_output=True, text=True, executable='/bin/bash'
            )
            self.logger.info(f"Successfully submitted jobs using sbatch for script {job_script_path}")
            # Capture and log the job array ID from the submission output
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Submitted job array with ID: {job_id}")
            self.track_job_progress(job_id)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during job submission process: {e}")
            raise


    def track_job_progress(self, job_id):
        """
        Tracks the progress of the submitted job array by periodically checking its status.
        Logs the current status and reports when the job array is completed.
        """
        self.logger.info(f"Tracking progress of job array with ID: {job_id}")
        start_time = time.time()
        check_interval = 30  # Check the job status every 30 seconds
        print_interval = 1 * 60  # Print job status every 1 minute
        last_print_time = start_time
        while True:
            result = subprocess.run(
                f'squeue --job {job_id} -h -o %T',
                shell=True, capture_output=True, text=True, executable='/bin/bash'
            )
            if result.returncode != 0:
                self.logger.error(f"Error checking job status for job ID {job_id}: {result.stderr.strip()}")
                break
            job_statuses = result.stdout.strip().split()
            if not job_statuses:
                self.logger.info(f"Job array {job_id} has completed.")
                break
            running_jobs = [status for status in ('PENDING', 'RUNNING', 'CONFIGURING') if status in job_statuses]
            current_time = time.time()
            if not running_jobs:
                self.logger.info(f"Job array {job_id} has completed.")
                break
            elif current_time - last_print_time >= print_interval:
                self.logger.info(f"Job array {job_id} is still running. Checking again in 5 mins...")
                last_print_time = current_time
            time.sleep(check_interval)
