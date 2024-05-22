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
    'compute_fixations': True})

root_data_dir = util.get_root_data_dir(params)
params.update({'root_data_dir': root_data_dir})

sorted_position_path_list, m1_gaze_positions, m2_gaze_positions = \
    filter_behav.get_gaze_positions_across_sessions(params)

#if params.get('compute_fixations', False):
    