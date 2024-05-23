#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:19:31 2024

@author: pg496
"""

import numpy as np
import os
import scipy

import util
import filter_behav
import filter_ephys
import plotter
import load_data

params = {}
params.update({
    'is_cluster': True,
    'use_parallel': False,
    'extract_postime_from_mat_files': True,
    'compute_fixations': True})

root_data_dir = util.get_root_data_dir(params)
params.update({'root_data_dir': root_data_dir})

if params.get('extract_postime_from_mat_files', False):
    sorted_position_path_list, m1_gaze_positions, m2_gaze_positions = \
        filter_behav.get_gaze_positions_across_sessions(params)
else:
    sorted_position_path_list, m1_gaze_positions, m2_gaze_positions = \
        load_data.get_combined_gaze_pos_lists(params)

params.update({'sorted_position_path_list': sorted_position_path_list,
               'm1_gaze_positions': m1_gaze_positions,
               'm2_gaze_positions': m2_gaze_positions})

if params.get('compute_fixations', False):
    x=1
    #fixations_m1, fixations_m2 = filter_behav.extract_fixations_for_both_monkeys(params)
else:
    x=2