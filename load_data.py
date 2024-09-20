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


def load_gaze_data_dict(gaze_data_file_path, missing_data_file_path):
    """
    Loads the gaze data dictionary and missing data paths from saved pickle files.
    Parameters:
    - gaze_data_file_path (str): Path to the saved gaze data dictionary file.
    - missing_data_file_path (str): Path to the saved missing data paths file.
    Returns:
    - gaze_data_dict (dict): The loaded gaze data dictionary.
    - missing_data_paths (list): The loaded list of missing data paths.
    """
    try:
        with open(gaze_data_file_path, 'rb') as f:
            gaze_data_dict = pickle.load(f)
        with open(missing_data_file_path, 'rb') as f:
            missing_data_paths = pickle.load(f)
        return gaze_data_dict, missing_data_paths
    except Exception as e:
        logger.error(f"Failed to load gaze data: {e}")
        raise









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



