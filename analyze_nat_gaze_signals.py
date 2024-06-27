#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:19:31 2024

@author: pg496
"""

import os
import re

import util
import filter_behav
import load_data
import fix_and_saccades

params = {}
params.update({
    'is_cluster': True,
    'use_parallel': True,
    'extract_postime_from_mat_files': False,
    'compute_fixations': False})

root_data_dir = util.get_root_data_dir(params)
params.update({'root_data_dir': root_data_dir})

if params.get('extract_postime_from_mat_files', False):
    sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
        filter_behav.get_gaze_timepos_across_sessions(params)
else:
    sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
        load_data.get_combined_gaze_pos_and_time_lists(params)

print(sorted_position_path_list)


pos_pattern = r"(\d{8})_position_(\d+).mat"
session_infos = [{'session_name': re.match(pos_pattern, os.path.basename(f)).group(1), 'run_number': int(re.match(pos_pattern, os.path.basename(f)).group(2))} for f in sorted_position_path_list]

# Extract fixations and saccades for both monkeys
fixations_m1, saccades_m1, fixations_m2, saccades_m2 = fix_and_saccades.extract_fixations_for_both_monkeys(params, m1_gaze_positions, m2_gaze_positions, time_vecs, session_infos)