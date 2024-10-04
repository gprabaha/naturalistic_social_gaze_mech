#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14::51:42 2024

@author: pg496
"""

import logging
import os
import multiprocessing
from datetime import datetime
from tqdm import tqdm

import util
import curate_data
import load_data
import fix_and_saccades
import plotter

import pdb



class DataManager:
    def __init__(self, params):
        self.setup_logger()
        self.params = params
        self.find_n_cores()
        self.initialize_class_objects()


    def setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)


    def find_n_cores(self):
        """Determine the number of CPU cores available, prioritizing SLURM if available."""
        try:
            slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
            num_cpus = int(slurm_cpus)
            self.logger.info(f"SLURM detected {num_cpus} CPUs")
        except Exception as e:
            self.logger.warning(f"Failed to detect cores with SLURM_CPUS_ON_NODE: {e}")
            num_cpus = None
        if num_cpus is None or num_cpus <= 1:
            num_cpus = multiprocessing.cpu_count()
            self.logger.info(f"Multiprocessing detected {num_cpus} CPUs")
        os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus)
        self.num_cpus = num_cpus
        self.params['num_cpus'] = num_cpus
        self.logger.info(f"NumExpr set to use {num_cpus} threads")


    def initialize_class_objects(self):
        """Initialize class object attributes."""
        self.gaze_data_df = None
        self.missing_data_paths = None
        self.nan_removed_gaze_data_df = None
        self.fixation_dict = None
        self.saccade_dict = None
        self.binary_behav_timeseries = None


    def populate_params_with_data_paths(self):
        self.params = curate_data.add_root_data_to_params(self.params)
        self.params = curate_data.add_processed_data_to_params(self.params)
        self.params = curate_data.add_raw_data_dir_to_params(self.params)
        self.params = curate_data.add_paths_to_all_data_files_to_params(self.params)
        self.params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(self.params)


    def get_data(self):
        """
        Loads gaze data into a dictionary format from the available position, time, and pupil size files.
        If the data already exists and `remake_gaze_data_dict` is False, it will load the data from a saved pickle file.
        Otherwise, it will recompute the gaze data using the defined function and save the result.
        This also handles the second output, which is the list of missing data paths.
        """
        # Define path to save/load the gaze data and the missing data paths
        gaze_data_file_path = os.path.join(self.params['processed_data_dir'], 'gaze_data_df.pkl')
        missing_data_paths_file_path = os.path.join(self.params['processed_data_dir'], 'missing_data_paths.pkl')
        # Use the compute_or_load_variables function to compute or load the gaze data and missing data paths
        self.gaze_data_df, self.missing_data_paths = util.compute_or_load_variables(
            curate_data.make_gaze_data_df,  # Function that computes the gaze data DataFrame and missing paths
            load_data.get_gaze_data_df,      # Function to load the saved gaze data and missing paths
            [gaze_data_file_path, missing_data_paths_file_path],  # Paths where the data will be saved/loaded
            'remake_gaze_data_dict',     # Parameter key to determine if we should remake or load the data
            self.params,                          # The params dictionary containing relevant settings
            self.params
        )


    def prune_data(self):
        processed_data_dir = self.params['processed_data_dir']
        nan_removed_gaze_data_file_path = os.path.join(processed_data_dir, 'nan_removed_gaze_data_df.pkl')
        # Use the compute_or_load_variables function to compute or load the gaze data
        self.nan_removed_gaze_data_df = util.compute_or_load_variables(
            curate_data.prune_nan_values_in_timeseries,    # Positional argument (compute_func)
            load_data.get_nan_removed_gaze_data_dict,         # Positional argument (load_func)
            nan_removed_gaze_data_file_path,                 # Positional argument (file_paths)
            'remake_nan_removed_gaze_data_dict',        # Positional argument (remake_flag_key)
            self.params,                                         # Positional argument (params)
            self.gaze_data_df                                         # This is passed as *args to compute_func
        )


    def analyze_behavior(self):
        # Path to where the fixation and saccade dictionaries are saved
        fixation_file_path = os.path.join(self.params['processed_data_dir'], 'fixation_dict.pkl')
        saccade_file_path = os.path.join(self.params['processed_data_dir'], 'saccade_dict.pkl')
        # !! Load and compute variables function also saves the variable that it computes !!
        self.fixation_dict, self.saccade_dict = util.compute_or_load_variables(
            fix_and_saccades.detect_fixations_and_saccades,  # Compute function
            load_data.load_fixation_and_saccade_dicts,       # Load function
            [fixation_file_path, saccade_file_path],         # File paths
            'remake_fix_and_sacc',                           # Remake flag key
            self.params,                                     # Params
            self.nan_removed_gaze_data_dict                  # Passed as the first positional argument
        )
        print('Fix dict:')
        util.print_dict_keys(self.fixation_dict)
        print('Sacc dict:')
        util.print_dict_keys(self.saccade_dict)
        self.binary_behav_timeseries = curate_data.generate_binary_behav_timeseries_dicts(self.fixation_dict, self.saccade_dict)
        util.print_dict_keys_and_values(self.binary_behav_timeseries)





    def plot_behavior(self):
        # Create the base directory under 'fixations_and_saccades' with today's date
        today_date = datetime.now().strftime('%Y-%m-%d')
        base_plot_dir = os.path.join(self.params['root_data_dir'], 'plots', 'fixations_and_saccades', today_date)
        # Call helper function to gather tasks
        tasks = plotter.gather_plotting_tasks(
            self.fixation_dict,
            self.saccade_dict,
            self.nan_removed_gaze_data_dict,
            base_plot_dir,
            self.params)
        # Execute tasks either in parallel or serial based on use_parallel flag
        if self.params.get('use_parallel', False):
            with multiprocessing.Pool(processes=self.params['num_cpus']) as pool:
                # Initialize the progress bar
                pbar = tqdm(total=len(tasks), desc="Plotting fix and saccades in parallel")
                # Submit tasks individually using apply_async and update progress after each task completes
                results = [pool.apply_async(plotter.plot_agent_behavior, args=task, callback=lambda _: pbar.update(1)) for task in tasks]
                # Ensure all tasks are completed
                for result in results:
                    result.wait()  # Wait for each task to complete
                # Close the progress bar when done
                pbar.close()
        else:
            for task in tqdm(tasks, desc="Plotting fix and saccades in serial"):
                plotter.plot_agent_behavior(*task)


    def run(self):
        self.populate_params_with_data_paths()
        self.get_data()
        self.prune_data()
        # self.analyze_behavior()
        # self.plot_behavior()

