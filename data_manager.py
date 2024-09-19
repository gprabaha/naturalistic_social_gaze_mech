#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14::51:42 2024

@author: pg496
"""

import logging

import util
import curate_data

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
        pdb.set_trace()
        
        # get_all_datafile_paths: should get paths to position files, time files, and pupil files (files same for m1 and m2). then ensure that the names and order matches
        # 

    # a method to load data that assumes that the filepaths are already there in the params
    # -> this should create some sort of a dictionary of positions with corresponding labels

    def get_data(self):
        """
        Loads gaze data into a dictionary format from the available position, time, and pupil size files.q
        """
        self.gaze_data_dict = curate_data.get_gaze_data_dict(self.params['data_file_paths'])


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