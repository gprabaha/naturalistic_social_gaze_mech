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


def get_combined_gaze_pos_lists(params):
    """
    Load combined gaze position lists.

    Args:
    - root_data_dir (str): Root directory containing data.

    Returns:
    - sorted_position_path_list (list): List of sorted file paths.
    - m1_positions (list): List of m1 gaze positions.
    - m2_positions (list): List of m2 gaze positions.
    """
    # Load sorted_position_path_list from the text file
    root_data_dir = params.get('root_data_dir')
    intermediates_dir = os.path.join(root_data_dir, 'intermediates')
    path_list_file = os.path.join(intermediates_dir, 'sorted_position_path_list.txt')
    with open(path_list_file, 'r') as f:
        sorted_position_path_list = f.read().splitlines()
        
    def load_arrays_from_npy(directory, filename_prefix):
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
    
    # Load m1_positions and m2_positions as lists of arrays
    m1_positions = load_arrays_from_npy(intermediates_dir, 'm1_positions')
    m2_positions = load_arrays_from_npy(intermediates_dir, 'm2_positions')
    return sorted_position_path_list, m1_positions, m2_positions

