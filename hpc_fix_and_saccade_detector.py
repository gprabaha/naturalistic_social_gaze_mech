import os
import subprocess
import logging
import time


class HPCFixAndSaccadeDetector:
    def __init__(self, params):
        self.params = params
        self.job_script_out_dir = './job_scripts/'
        self.logger = logging.getLogger(__name__)
        self.python_script_path = './run_single_fix_and_saccade_detection_task.py'  # Path to the new Python script

    def generate_job_file(self, tasks, params_file_path):
        """
        Generates a job file for submitting array jobs, with each task processing one run.
        """
        job_file_path = os.path.join(self.job_script_out_dir, f'{self.params["agent"]}_joblist.txt')
        os.makedirs(self.job_script_out_dir, exist_ok=True)

        with open(job_file_path, 'w') as file:
            for task in tasks:
                task_key = f"{task[0]},{task[1]},{task[2]},{task[3]}"
                command = (
                    f"module load miniconda; conda deactivate; "
                    f"conda activate gaze_otnal; "
                    f"python {self.python_script_path} {task_key} {params_file_path}"
                )
                file.write(command + "\n")
        self.logger.info(f"Generated job file at {job_file_path}")
        return job_file_path

    def submit_job_array(self, job_file_path):
        """
        Submits the generated job file as a job array to the cluster.
        """
        try:
            job_script_path = os.path.join(self.job_script_out_dir, f'dsq-joblist_fixations_{self.params["agent"]}.sh')
            partition = 'day' if self.params['is_grace'] else 'psych_day'
            subprocess.run(
                f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} '
                f'-o {self.job_script_out_dir} --status-dir {self.job_script_out_dir} --partition {partition} '
                f'--cpus-per-task 4 --mem-per-cpu 8000 -t 4:00:00 --mail-type FAIL',
                shell=True, check=True, executable='/bin/bash')
            self.logger.info(f"Successfully generated the dSQ job script ({self.params['agent']})")
            if not os.path.isfile(job_script_path):
                self.logger.error(f"No job script found at {job_script_path}.")
                return
            self.logger.info(f"Using dSQ job script: {job_script_path}")
            result = subprocess.run(
                f'sbatch --job-name=fix_dsq_{self.params['agent']} --output={self.job_script_out_dir}/fixation_session_{self.params['agent']}_%a.out '
                f'--error={self.job_script_out_dir}/fixation_session_{self.params['agent']}_%a.err {job_script_path}',
                shell=True, check=True, capture_output=True, text=True, executable='/bin/bash')
            self.logger.info(f"Successfully submitted jobs using sbatch for script {job_script_path}")
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Submitted job array with ID: {job_id} ({self.params['agent']})")
            self.track_job_progress(job_id)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during job submission process: {e}")
            raise

    def track_job_progress(self, job_id):
        """
        Tracks the progress of the submitted job array by periodically checking its status.
        Logs the current status and reports when the job array is completed.
        """
        self.logger.info(f"Tracking progress of job array with ID: {job_id} ({self.params['agent']})")
        start_time = time.time()
        check_interval = 30  # Check the job status every 30 seconds
        print_interval = 5 * 60  # Print job status every 5 minutes
        last_print_time = start_time
        while True:
            result = subprocess.run(
                f'squeue --job {job_id} -h -o %T',
                shell=True, capture_output=True, text=True, executable='/bin/bash'
            )
            if result.returncode != 0:
                self.logger.error(f"Error checking job status for job ID {job_id} ({self.params['agent']}): {result.stderr.strip()}")
                break
            job_statuses = result.stdout.strip().split()
            if not job_statuses:
                self.logger.info(f"Job array {job_id} ({self.params['agent']}) has completed.")
                break
            running_jobs = [status for status in ('PENDING', 'RUNNING', 'CONFIGURING') if status in job_statuses]
            current_time = time.time()
            if not running_jobs:
                self.logger.info(f"Job array {job_id} ({self.params['agent']}) has completed.")
                break
            elif current_time - last_print_time >= print_interval:
                self.logger.info(f"Job array {job_id} ({self.params['agent']}) is still running. Checking again in 5 mins...")
                last_print_time = current_time
            time.sleep(check_interval)
