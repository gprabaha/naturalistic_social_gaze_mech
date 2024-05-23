#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:56:31 2024

@author: pg496
"""

import os
import re
import scipy.io
import numpy as np


def load_mat_from_path(path):
    return scipy.io.loadmat(path)


def get_combined_gaze_pos_and_time_lists(params):
    """
    Load combined gaze position and time lists.
    Args:
    - params (dict): Dictionary containing parameters.
    Returns:
    - sorted_position_path_list (list): List of sorted file paths.
    - m1_positions (list): List of m1 gaze positions.
    - m2_positions (list): List of m2 gaze positions.
    - time_vectors (list): List of time vectors.
    """
    root_data_dir = params.get('root_data_dir')
    intermediates_dir = os.path.join(root_data_dir, 'intermediates')
    # Load sorted position and time file lists
    sorted_position_path_list_file = os.path.join(intermediates_dir, 'sorted_position_path_list.txt')
    sorted_time_path_list_file = os.path.join(intermediates_dir, 'sorted_time_path_list.txt')
    with open(sorted_position_path_list_file, 'r') as f:
        sorted_position_path_list = f.read().splitlines()
    with open(sorted_time_path_list_file, 'r') as f:
        sorted_time_path_list = f.read().splitlines()
    # Load m1_positions, m2_positions, and time_vectors as lists of arrays
    m1_positions = load_arrays_from_multiple_npy(intermediates_dir, 'm1_positions')
    m2_positions = load_arrays_from_multiple_npy(intermediates_dir, 'm2_positions')
    time_vectors = load_arrays_from_multiple_npy(intermediates_dir, 'time_vectors')
    return sorted_position_path_list, m1_positions, m2_positions, sorted_time_path_list, time_vectors



def load_arrays_from_multiple_npy(directory, filename_prefix):
    """
    Load arrays from .npy files with preserved list structure.
    Args:
    - directory (str): Directory containing the files.
    - filename_prefix (str): Prefix for the input filenames.
    Returns:
    - arrays (list): List of loaded arrays.
    """
    arrays = []
    i = 0
    while True:
        input_file = os.path.join(directory, f"{filename_prefix}_{i}.npy")
        if not os.path.exists(input_file):
            break
        arrays.append(np.load(input_file))
        i += 1
    return arrays



