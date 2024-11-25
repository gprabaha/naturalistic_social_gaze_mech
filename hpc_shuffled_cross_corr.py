import os
import random
import subprocess
import logging
import time

import pdb

class HPCShuffledCrossCorr:
    def __init__(self, params):
        self.params = params
        self.job_script_out_dir = './job_scripts/'
        self.logger = logging.getLogger(__name__)
        self.python_script_path = 'run_shuffled_cross_corr_for_one_job_in_array.py'  # Path to the Python script


    def generate_job_file(self, groups, binary_timeseries_file_path, num_cpus, num_shuffles):
        """
        Generates a job file for submitting array jobs, with each task processing one group.
        If try_using_single_run is True, only one random group is selected for testing.
        """
        job_file_path = os.path.join(self.job_script_out_dir, 'joblist_shuffled_cross_corr.txt')
        os.makedirs(self.job_script_out_dir, exist_ok=True)
        env_name = 'gaze_processing'
        # If try_using_single_run is True, randomly select one group
        if self.params.get('try_using_single_run', False):
            self.logger.info("Using a single random group for testing.")
            groups = [random.choice(list(groups))]
        with open(job_file_path, 'w') as file:
            for group_keys in groups:
                group_str = f'"{group_keys}"'
                command = (
                    f"module load miniconda; "
                    f"conda activate {env_name}; "
                    f"python {self.python_script_path} --group_keys {group_str} "
                    f"--binary_timeseries_file {binary_timeseries_file_path} "
                    f"--output_dir {os.path.join(self.params.get('processed_data_dir'), 'crosscorr_chunk_outputs_shuffled')} "
                    f"--num_cpus {num_cpus} "
                    f"--shuffle_count {num_shuffles}"
                )
                file.write(command + "\n")
        self.logger.info(f"Generated job file at {job_file_path}")
        return job_file_path


    def submit_job_array(self, job_file_path, num_cpus):
        """
        Submits the generated job file as a job array to the cluster using dSQ.
        """
        try:
            job_script_path = os.path.join(self.job_script_out_dir, 'dsq-joblist_shuffled_cross_corr.sh')
            partition = 'psych_day'
            # Generate the dSQ job script
            subprocess.run(
                f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} '
                f'-o {self.job_script_out_dir} --status-dir {self.job_script_out_dir} --partition {partition} '
                f'--cpus-per-task {num_cpus} --mem-per-cpu 10G -t 00:30:00',
                shell=True, check=True, executable='/bin/bash'
            )
            self.logger.info("Successfully generated the dSQ job script.")
            if not os.path.isfile(job_script_path):
                self.logger.error(f"No job script found at {job_script_path}.")
                return
            self.logger.info(f"Using dSQ job script: {job_script_path}")
            result = subprocess.run(
                f'sbatch --job-name=dsq_shuffled_cross_corr '
                f'--output={self.job_script_out_dir}/shuffled_cross_corr_%a.out '
                f'--error={self.job_script_out_dir}/shuffled_cross_corr_%a.err '
                f'{job_script_path}',
                shell=True, check=True, capture_output=True, text=True, executable='/bin/bash'
            )
            self.logger.info(f"Successfully submitted jobs using sbatch for script {job_script_path}")
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Submitted job array with ID: {job_id}")
            self.track_job_progress(job_id)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during job submission process: {e}")
            raise


    def track_job_progress(self, job_id):
        """
        Tracks the progress of the submitted job array by periodically checking its status.
        """
        self.logger.info(f"Tracking progress of job array with ID: {job_id}")
        start_time = time.time()
        check_interval = 30
        print_interval = 60
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
                self.logger.info(f"Job array {job_id} is still running. Checking again in 1 minute...")
                last_print_time = current_time
            time.sleep(check_interval)
