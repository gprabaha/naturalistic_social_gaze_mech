#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:04:31 2024

@author: pg496
"""

import re
import os
import scipy
import numpy as np

import load_data

def get_gaze_positions_across_sessions(params):
    """
    Get gaze positions across sessions.

    Args:
    - params (dict): Dictionary containing parameters.

    Returns:
    - sorted_position_path_list (list): List of sorted file paths.
    - m1_positions (list): List of m1 gaze positions.
    - m2_positions (list): List of m2 gaze positions.
    """
    root_data_dir = params.get('root_data_dir')
    intermediates_dir = os.path.join(root_data_dir, 'intermediates')
    os.makedirs(intermediates_dir, exist_ok=True)

    rel_path_to_positions = 'eyetracking/aligned_raw_samples/position'
    path_to_positions = os.path.join(root_data_dir, rel_path_to_positions)
    # Step 1: Get a list of all .mat files in the directory
    mat_files = [f for f in os.listdir(path_to_positions) if f.endswith('.mat')]
    
    # Step 2: Sort the files based on date and run number
    def sort_key(filename):
        match = re.match(r"(\d{8})_positions_(\d+).mat", filename)
        if match:
            date_str, run_str = match.groups()
            date_key = date_str
            run_key = int(run_str)
            return (date_key, run_key)
        return (filename, 0)
    
    mat_files_sorted = sorted(mat_files, key=sort_key)
    sorted_position_path_list = [os.path.join(path_to_positions, f) for f in mat_files_sorted]
    # Step 3: Initialize lists for m1 and m2 gaze positions
    m1_positions = []
    m2_positions = []
    # Step 4: Iterate through the sorted files and extract the data
    for mat_file in sorted_position_path_list:
        mat_data = load_data.load_mat_from_path(mat_file)
        if 'var' in mat_data:
            aligned_positions = mat_data['var'][0][0]
            if 'm1' in aligned_positions.dtype.names and 'm2' in aligned_positions.dtype.names:
                m1_positions.append(aligned_positions['m1'].tolist())
                m2_positions.append(aligned_positions['m2'].tolist())
                
        def save_arrays_as_npy(directory, arrays, filename_prefix):
            """
            Save arrays as .npy files with preserved list structure.
            Args:
            - directory (str): Directory to save the files.
            - arrays (list): List of arrays to save.
            - filename_prefix (str): Prefix for the output filenames.
            """
            for i, array in enumerate(arrays):
                output_file = os.path.join(directory, f"{filename_prefix}_{i}.npy")
                np.save(output_file, array)

        # Step 5: Save m1_positions and m2_positions as .npy files
        save_arrays_as_npy(intermediates_dir, m1_positions, 'm1_positions')
        save_arrays_as_npy(intermediates_dir, m2_positions, 'm2_positions')
        # Step 6: Save sorted_position_path_list as a text file
        path_list_file = os.path.join(intermediates_dir, 'sorted_position_path_list.txt')
        with open(path_list_file, 'w') as f:
            f.write('\n'.join(sorted_position_path_list))
        # Step 7: Return the sorted file paths and extracted positions
        return sorted_position_path_list, m1_positions, m2_positions