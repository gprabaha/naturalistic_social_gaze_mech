#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:23:31 2024

@author: pg496
"""

import os
import re
import numpy as np

from math import sqrt

import pdb


def get_root_data_dir(params):
    """
    Returns the root data directory based on whether it's running on a cluster or not.
    Parameters:
    - params (dict): Dictionary containing parameters.
    Returns:
    - root_data_dir (str): Root data directory path.
    """
    is_cluster = params['is_cluster']
    return "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/" if is_cluster \
        else "/Volumes/Stash/changlab/social_gaze"


def get_sorted_files(directory, pattern):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mat')]

    def sort_key(filepath):
        basename = os.path.basename(filepath)  # Extract the filename from the path
        match = re.match(pattern, basename)
        if match:
            date_str, run_str = match.groups()
            date_key = date_str
            run_key = int(run_str)
            return (date_key, run_key)
        return None

    # Create a list of tuples (file, sort_key) where sort_key is not None
    files_with_keys = [(f, sort_key(f)) for f in files]
    files_with_keys = [item for item in files_with_keys if item[1] is not None]

    # Sort the list of tuples by the sort_key
    sorted_files_with_keys = sorted(files_with_keys, key=lambda item: item[1])

    # Extract the sorted files from the sorted list of tuples
    sorted_files = [item[0] for item in sorted_files_with_keys]

    return sorted_files




def save_arrays_as_npy(directory, arrays, filename_prefix):
    for i, array in enumerate(arrays):
        output_file = os.path.join(directory, f"{filename_prefix}_{i}.npy")
        np.save(output_file, array)


def filter_none_entries(*lists):
    filtered_lists = []
    for lst in lists:
        filtered_lists.append([item for item in lst if item is not None])
    return filtered_lists


def synchronize_file_lists(time_files, pos_files, m1_positions, m2_positions, time_vectors, pattern):
    """
    Ensure the list of position files matches the list of time files.
    Args:
    - time_files (list): List of time file paths.
    - pos_files (list): List of position file paths.
    - m1_positions (list): List of m1 gaze positions.
    - m2_positions (list): List of m2 gaze positions.
    - time_vectors (list): List of time vectors.
    Returns:
    - (list, list, list, list): Synchronized time_files, pos_files, m1_positions, m2_positions, time_vectors.
    """
    # Debug prints to check the input types
    for i, path in enumerate(pos_files):
        if not isinstance(path, (str, bytes, os.PathLike)):
            print(f"Error in pos_files at index {i}: {path} is of type {type(path)}")

    for i, path in enumerate(time_files):
        if not isinstance(path, (str, bytes, os.PathLike)):
            print(f"Error in time_files at index {i}: {path} is of type {type(path)}")

    # Creating dictionaries with filenames as keys
    pos_dict = {os.path.basename(path): (path, m1_pos, m2_pos) for path, m1_pos, m2_pos in zip(pos_files, m1_positions, m2_positions)}
    time_dict = {os.path.basename(path): (path, t) for path, t in zip(time_files, time_vectors)}
    
    def sort_key(filepath):
        basename = os.path.basename(filepath)  # Extract the filename from the path
        match = re.match(pattern, basename)
        if match:
            date_str, run_str = match.groups()
            date_key = date_str
            run_key = int(run_str)
            return (date_key, run_key)
        return None
    
    # Identifying common filenames
    common_filenames = set(pos_dict.keys()).intersection(time_dict.keys())
    # Create a list of tuples (file, sort_key) where sort_key is not None
    files_with_keys = [(f, sort_key(f)) for f in common_filenames]
    files_with_keys = [item for item in files_with_keys if item[1] is not None]
    # Sort the list of tuples by the sort_key
    sorted_files_with_keys = sorted(files_with_keys, key=lambda item: item[1])
    # Extract the sorted files from the sorted list of tuples
    common_filenames_sorted = [item[0] for item in sorted_files_with_keys]

    # Initializing synchronized lists
    synchronized_time_files = []
    synchronized_pos_files = []
    synchronized_m1_positions = []
    synchronized_m2_positions = []
    synchronized_time_vectors = []

    # Collecting synchronized data
    for filename in common_filenames_sorted:
        synchronized_time_files.append(time_dict[filename][0])
        synchronized_pos_files.append(pos_dict[filename][0])
        synchronized_m1_positions.append(pos_dict[filename][1])
        synchronized_m2_positions.append(pos_dict[filename][2])
        synchronized_time_vectors.append(time_dict[filename][1])

    return synchronized_time_files, synchronized_pos_files, synchronized_m1_positions, synchronized_m2_positions, synchronized_time_vectors


def remove_nans(positions, time_vec):
    # Create a boolean mask where time_vec is not NaN
    mask = ~np.isnan(time_vec).flatten()

    # Apply the mask to both positions and time_vec
    nan_removed_positions = positions[mask]
    nan_removed_time_vec = time_vec[mask]

    return nan_removed_positions, nan_removed_time_vec


def distance2p(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    Parameters:
    - point1 (tuple): First point coordinates.
    - point2 (tuple): Second point coordinates.
    Returns:
    - dist (float): Euclidean distance.
    """
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

