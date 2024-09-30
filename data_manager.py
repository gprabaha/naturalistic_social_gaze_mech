#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14::51:42 2024

@author: pg496
"""

import logging
import os
import multiprocessing

import util
import curate_data
import load_data
import fix_and_saccades

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
        self.gaze_data_dict = None
        self.empty_gaze_dict_paths = None
        self.nan_removed_gaze_data_dict = None

        self.fixation_dict = None
        self.saccade_dict = None


    def populate_params_with_data_paths(self):
        self.params = curate_data.add_root_data_to_params(self.params)
        self.params = curate_data.add_processed_data_to_params(self.params)
        self.params = curate_data.add_raw_data_dir_to_params(self.params)
        self.params = curate_data.add_paths_to_all_data_files_to_params(self.params)
        self.params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(self.params)


    def get_data(self):
        """
        Loads gaze data into a dictionary format from the available position, time, and pupil size files.
        """
        # Define paths to save/load the variables
        processed_data_dir = self.params['processed_data_dir']
        gaze_data_file_path = os.path.join(processed_data_dir, 'gaze_data_dict.pkl')
        # Use the compute_or_load_variables function to compute or load the gaze data
        # !! Load and compute variables function also saves the variable that it computes !!
        self.gaze_data_dict = util.compute_or_load_variables(
            compute_func=curate_data.make_gaze_data_dict,
            load_func=load_data.get_gaze_data_dict,  # Function to load the data, to be implemented next
            file_paths=gaze_data_file_path,
            remake_flag_key='remake_gaze_data_dict',
            params=self.params  # Pass the params dictionary
        )


    def prune_data(self):
        self.gaze_data_dict, self.missing_data_in_dict = curate_data.clean_and_log_missing_dict_leaves(self.gaze_data_dict)

        processed_data_dir = self.params['processed_data_dir']
        nan_removed_gaze_data_file_path = os.path.join(processed_data_dir, 'nan_removed_gaze_data_dict.pkl')
        # Use the compute_or_load_variables function to compute or load the gaze data
        # !! Load and compute variables function also saves the variable that it computes !!
        self.nan_removed_gaze_data_dict = util.compute_or_load_variables(
            curate_data.prune_nan_values_in_timeseries,  # Positional argument (compute_func)
            load_data.get_nan_removed_gaze_data_dict,    # Positional argument (load_func)
            nan_removed_gaze_data_file_path,             # Positional argument (file_paths)
            'remake_nan_removed_gaze_data_dict',         # Positional argument (remake_flag_key)
            self.params,                                 # Positional argument (params)
            self.gaze_data_dict                          # Positional argument (additional arguments)
        )


    def analyze_behavior(self):
        # Detect fixations and saccades for m1
        self.fixation_dict, self.saccade_dict = fix_and_saccades.detect_fixations_and_saccades(
            self.nan_removed_gaze_data_dict, params=self.params
        )


    def run(self):
        self.populate_params_with_data_paths()
        self.get_data()
        self.prune_data()
        pdb.set_trace()
        self.analyze_behavior()




# if params.get('extract_postime_from_mat_files', False):
#     sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
#         filter_behav.get_gaze_timepos_across_sessions(params)
# else:
#     sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
#         load_data.get_combined_gaze_pos_and_time_lists(params)

# print(sorted_position_path_list)


# pos_pattern = r"(\d{8})_position_(\d+).mat"
# session_infos = [{'session_name': re.match(pos_pattern, os.path.basename(f)).group(1), 'run_number': int(re.match(pos_pattern, os.path.basename(f)).group(2))} for f in sorted_position_path_list]

# # Extract fixations and saccades for both monkeys
# fixations_m1, saccades_m1, fixations_m2, saccades_m2 = fix_and_saccades.extract_fixations_for_both_monkeys(params, m1_gaze_positions, m2_gaze_positions, time_vecs, session_infos)