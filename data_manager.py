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


def print_dict_structure(d, indent=0):
    """
    Recursively prints the structure of a nested dictionary without printing the values.

    :param d: The dictionary to explore.
    :param indent: The current level of indentation (used for nested dictionaries).
    """
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + str(key) + ':')
            print_dict_structure(value, indent + 1)
    elif isinstance(d, list):
        print('  ' * indent + '[List]')
    else:
        print('  ' * indent + '[Value]')


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


    def populate_params_with_data_paths(self):
        self.params = curate_data.add_root_data_to_params(self.params)
        self.params = curate_data.add_processed_data_to_params(self.params)
        self.params = curate_data.add_raw_data_dir_to_params(self.params)
        self.params = curate_data.add_paths_to_all_data_files_to_params(self.params)
        self.params = curate_data.prune_data_file_paths(self.params)
        pdb.set_trace()
        return 0


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


    def prune_nan_values_in_timeseries(self):
        """
        Prunes NaN values from the time series in the gaze data dictionary and 
        adjusts positions and pupil_size for m1 and m2 (if present) accordingly.
        The pruned dictionary is stored in `self.nan_removed_gaze_data_dict`.
        """
        # Create a copy of the gaze data dictionary to store the pruned version
        self.nan_removed_gaze_data_dict = {}
        # Iterate over the original gaze data dictionary
        for session, session_dict in self.gaze_data_dict.items():
            pruned_session_dict = {}
            for interaction_type, interaction_dict in session_dict.items():
                pruned_interaction_dict = {}
                for run, run_dict in interaction_dict.items():
                    # Extract the time series
                    time_series = run_dict.get('neural_timeline')
                    if time_series is not None:
                        # Prune NaN values and adjust corresponding timeseries using the helper function
                        pruned_positions, pruned_pupil_size, pruned_time_series = util.prune_nans_in_specific_timeseries(
                            time_series,
                            run_dict.get('positions', {}),
                            run_dict.get('pupil_size', {})
                        )
                        # Create a new run dictionary with pruned data
                        pruned_run_dict = {
                            'positions': pruned_positions,
                            'pupil_size': pruned_pupil_size,
                            'time': pruned_time_series
                        }
                        pruned_interaction_dict[run] = pruned_run_dict
                pruned_session_dict[interaction_type] = pruned_interaction_dict
            self.nan_removed_gaze_data_dict[session] = pruned_session_dict
        return self.nan_removed_gaze_data_dict


    def analyze_behavior(self):
        self.nan_removed_gaze_data_dict = self.prune_nan_values_in_timeseries()
        # pdb.set_trace()
        # return 0


    def run(self):
        self.populate_params_with_data_paths()
        self.get_data()
        self.analyze_behavior()
        # Use the function on your dictionary
        print_dict_structure(self.nan_removed_gaze_data_dict)



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