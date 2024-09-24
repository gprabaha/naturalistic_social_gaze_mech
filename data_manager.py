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
import fix_and_saccades

import pdb


class DataManager:
    def __init__(self, params):
        self.params = params
        self.setup_logger()
        self.initialize_class_objects()


    def setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)


    def initialize_class_objects(self):
        self.gaze_data_dict = None
        self.empty_gaze_dict_paths = None
        self.nan_removed_gaze_data_dict = None

        self.fixation_df_m1 = None
        self.fixation_df_m2 = None
        self.saccade_df_m1 = None
        self.saccade_df_m2 = None
        self.microsaccade_df_m1 = None
        self.microsaccade_df_m2 = None


    def populate_params_with_data_paths(self):
        self.params = curate_data.add_root_data_to_params(self.params)
        self.params = curate_data.add_processed_data_to_params(self.params)
        self.params = curate_data.add_raw_data_dir_to_params(self.params)
        self.params = curate_data.add_paths_to_all_data_files_to_params(self.params)
        self.params = curate_data.prune_data_file_paths(self.params)


    def get_data(self):
        """
        Loads gaze data into a dictionary format from the available position, time, and pupil size files.
        """
        # Define paths to save/load the variables
        processed_data_dir = self.params['processed_data_dir']
        gaze_data_file_path = os.path.join(processed_data_dir, 'gaze_data_dict.pkl')
        missing_data_file_path = os.path.join(processed_data_dir, 'missing_data_dict_paths.pkl')
        # Use the compute_or_load_variables function to compute or load the gaze data
        self.gaze_data_dict, self.empty_gaze_dict_paths = util.compute_or_load_variables(
            compute_func=curate_data.make_gaze_data_dict,
            load_func=load_data.get_gaze_data_dict,  # Function to load the data, to be implemented next
            file_paths=[gaze_data_file_path, missing_data_file_path],
            remake_flag_key='remake_gaze_data_dict',
            params=self.params  # Pass the params dictionary
        )


    def analyze_behavior(self):
        self.nan_removed_gaze_data_dict = util.prune_nan_values_in_timeseries(self.gaze_data_dict)
        # Detect fixations and saccades for m1
        self.fixation_dict_m1, self.saccade_dict_m1 = fix_and_saccades.detect_fixations_and_saccades(
            self.nan_removed_gaze_data_dict, agent='m1', params=self.params
        )
        # Detect fixations and saccades for m2
        self.fixation_dict_m2, self.saccade_dict_m2 = fix_and_saccades.detect_fixations_and_saccades(
            self.nan_removed_gaze_data_dict, agent='m2', params=self.params
        )


    def run(self):
        self.populate_params_with_data_paths()
        self.get_data()
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