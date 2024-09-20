#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14::51:42 2024

@author: pg496
"""

import logging
import os
import pickle

import util
import curate_data
import load_data

import pdb


class DataManager:
    def __init__(self, params):
        self.params = params
        self.setup_logger()
        self.initialize_params()


    def setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)


    def initialize_params(self):
        self.params = util.add_root_data_to_params(self.params)
        self.params = util.add_processed_data_to_params(self.params)
        self.params = util.add_raw_data_dir_to_params(self.params)
        self.params = util.add_paths_to_all_data_files_to_params(self.params)
        self.params = util.prune_data_file_paths(self.params)

        self.gaze_data_dict = None
        self.empty_gaze_dict_paths = None



    def compute_or_load_variables(self, compute_func, load_func, file_paths, remake_flag_key, *args, **kwargs):
        """
        Generic method to manage compute vs. load actions for various data like gaze, fixations, saccades, etc.
        Parameters:
        - compute_func (function): The function that computes the data.
        - load_func (function): The function that loads the data from saved files.
        - file_paths (list): List of file paths where each variable will be saved or loaded from.
        - remake_flag_key (str): The key in self.params to check whether to compute or load (e.g., 'remake_gaze_data_dict').
        - args, kwargs: Additional arguments to pass to the compute_func.
        Returns:
        - A list of variables, either loaded from files or computed.
        """
        remake_flag = self.params.get(remake_flag_key, True)  # Check the corresponding remake flag
        if remake_flag:
            # Compute the data
            computed_vars = compute_func(*args, **kwargs)
            # Save each computed variable to its corresponding file path
            for file_path, var in zip(file_paths, computed_vars):
                with open(file_path, 'wb') as f:
                    pickle.dump(var, f)
            return computed_vars
        else:
            # Load the data using the provided load function
            loaded_vars = load_func(*file_paths)
            return loaded_vars


    def get_data(self):
        """
        Loads gaze data into a dictionary format from the available position, time, and pupil size files.
        """
        # Define paths to save/load the variables
        processed_data_dir = self.params['processed_data_dir']
        gaze_data_file_path = os.path.join(processed_data_dir, 'gaze_data_dict.pkl')
        missing_data_file_path = os.path.join(processed_data_dir, 'missing_data_dict_paths.pkl')
        # Use the manage_data method to compute or load the gaze data
        self.gaze_data_dict, self.empty_gaze_dict_paths = self.compute_or_load_variables(
            compute_func = curate_data.get_gaze_data_dict,
            load_func = load_data.load_gaze_data_dict,  # Function to load the data, to be implemented next
            file_paths = [gaze_data_file_path, missing_data_file_path],
            remake_flag_key = 'remake_gaze_data_dict',
            params = self.params  # Pass additional required parameters
        )


    def run(self):
        self.get_data()



params = {}
params.update({
    'processed_data_dir': 'intermediates',
    'is_cluster': True,
    'use_parallel': True,
    'extract_postime_from_mat_files': False,
    'compute_fixations': False,
    'fixation_detection_method': 'eye_mvm',
    'sampling_rate': 0.001,
    'vel_thresh': 10
    })




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