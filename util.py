#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:23:31 2024

@author: pg496
"""

import logging
import os
import re
import numpy as np
import pickle
from math import sqrt

import pdb


# Set up a logger for this module
logger = logging.getLogger(__name__)

def add_root_data_to_params(params):
    """
    Sets the root data directory based on cluster and Grace settings.
    Parameters:
    - params (dict): Dictionary containing configuration parameters, including flags for cluster and Grace.
    Returns:
    - params (dict): Updated dictionary with the 'root_data_dir' field added.
    """
    logger.info("Setting root data directory based on cluster and Grace settings.")
    if params.get('is_cluster', True):
        root_data_dir = "/gpfs/gibbs/project/chang/pg496/data_dir/social_gaze/" if params.get('is_grace', False) \
                        else "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/"
    else:
        root_data_dir = "/Volumes/Stash/changlab/sorted_neural_data/social_gaze/"
    params['root_data_dir'] = root_data_dir
    logger.info(f"Root data directory set to: {root_data_dir}")
    return params


def add_processed_data_to_params(params):
    """
    Adds the processed data directory path to the parameters.
    Parameters:
    - params (dict): Dictionary containing configuration parameters with 'root_data_dir' defined.
    Returns:
    - params (dict): Updated dictionary with 'processed_data_dir' field added.
    """
    root_data_dir = params.get('root_data_dir')
    processed_data_dir = os.path.join(root_data_dir, 'intermediates')
    params.update({'processed_data_dir': processed_data_dir})
    logger.info(f"Processed data directory set to: {processed_data_dir}")
    return params


def add_raw_data_dir_to_params(params):
    """
    Adds paths to raw data directories for positions, neural timeline, and pupil size.
    Parameters:
    - params (dict): Dictionary containing configuration parameters with 'root_data_dir' defined.
    Returns:
    - params (dict): Updated dictionary with 'positions_dir', 'neural_timeline_dir', and 'pupil_size_dir' fields added.
    """
    root_data_dir = params.get('root_data_dir')
    path_to_positions = os.path.join(root_data_dir, 'eyetracking/aligned_raw_samples/position')
    path_to_time_vecs = os.path.join(root_data_dir, 'eyetracking/aligned_raw_samples/time')
    path_to_pupil_vecs = os.path.join(root_data_dir, 'eyetracking/aligned_raw_samples/pupil_size')
    params['positions_dir'] = path_to_positions
    params['neural_timeline_dir'] = path_to_time_vecs
    params['pupil_size_dir'] = path_to_pupil_vecs
    logger.info("Raw data directories added to params.")
    return params


def add_paths_to_all_data_files_to_params(params):
    """
    Populates the paths to data files categorized by session, interaction type, and run number.
    Parameters:
    - params (dict): Dictionary containing paths to 'positions_dir', 'neural_timeline_dir', and 'pupil_size_dir'.
    Returns:
    - params (dict): Updated dictionary with 'data_file_paths' field, which contains paths categorized by session
      and interaction type.
    """
    # Define directories
    directories = {
        'positions': params['positions_dir'],
        'neural_timeline': params['neural_timeline_dir'],
        'pupil_size': params['pupil_size_dir']
    }
    # Initialize data structure to store file paths
    paths_dict = {'positions': {}, 'neural_timeline': {}, 'pupil_size': {}}
    # Define regex to extract session name (date) and run number
    file_pattern = re.compile(r'(\d{8})_(position|dot)_(\d+)\.mat')
    logger.info("Populating paths to data files.")
    # Iterate over each directory
    for key, dir_path in directories.items():
        logger.info(f"Processing directory: {dir_path}")
        try:
            for filename in os.listdir(dir_path):
                match = file_pattern.match(filename)
                if match:
                    session_name, file_type, run_number = match.groups()
                    run_number = int(run_number)
                    # Initialize session dictionary if not already
                    if session_name not in paths_dict[key]:
                        paths_dict[key][session_name] = {
                            'interactive': {},
                            'non_interactive': {}
                        }
                    # Determine file category and assign path
                    if file_type == 'position':
                        paths_dict[key][session_name]['interactive'][run_number] = os.path.join(dir_path, filename)
                    elif file_type == 'dot':
                        paths_dict[key][session_name]['non_interactive'][run_number] = os.path.join(dir_path, filename)
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")
    # Add legend explaining the dictionary structure
    paths_dict['legend'] = {
        'positions': 'Paths for positions data, categorized by session, then by interactive/non_interactive.',
        'neural_timeline': 'Paths for neural timeline data, categorized by session, then by interactive/non_interactive.',
        'pupil_size': 'Paths for pupil size data, categorized by session, then by interactive/non_interactive.',
        'session_name': 'Top-level keys representing session dates (8 digits).',
        'interactive': 'Files with "position" in the name, grouped under run number keys.',
        'non_interactive': 'Files with "dot" in the name, grouped under run number keys.'
    }
    logger.info("Paths to all data files populated successfully.")
    # Update params with the generated paths dictionary
    params['data_file_paths'] = paths_dict
    return params


def prune_data_file_paths(params):
    """
    Prunes the data file paths to ensure that positions, neural timeline, and pupil size all have the same set
    of file names. Files present in one folder but not the others are discarded and recorded.
    Parameters:
    - params (dict): Dictionary containing 'data_file_paths' with paths categorized by session, interaction type, 
      and run number.
    Returns:
    - params (dict): Updated dictionary with pruned 'data_file_paths' and a new 'discarded_paths' field 
      that records paths of discarded files.
    """
    logger.info("Pruning data file paths to ensure consistency across data types.")
    # Extract the data paths dictionary from params
    paths_dict = params.get('data_file_paths', {})
    discarded_paths = {'positions': {}, 'neural_timeline': {}, 'pupil_size': {}}
    # Get the keys (positions, neural_timeline, pupil_size) for processing
    data_keys = ['positions', 'neural_timeline', 'pupil_size']
    # Find common sessions across all data types
    common_sessions = set(paths_dict['positions'].keys())
    for key in data_keys[1:]:
        common_sessions.intersection_update(paths_dict[key].keys())
    # Iterate through common sessions to find common runs and prune mismatches
    for session in common_sessions:
        # Gather all run numbers present in each data type for this session
        runs_per_type = {key: set(paths_dict[key][session]['interactive'].keys()) | set(paths_dict[key][session]['non_interactive'].keys()) for key in data_keys}
        # Find the intersection of runs that exist in all data types
        common_runs = set.intersection(*runs_per_type.values())
        # Prune files not in the common runs for each data type
        for key in data_keys:
            # Check interactive runs
            interactive_runs = set(paths_dict[key][session]['interactive'].keys())
            non_interactive_runs = set(paths_dict[key][session]['non_interactive'].keys())
            # Identify runs to discard
            discard_interactive = interactive_runs - common_runs
            discard_non_interactive = non_interactive_runs - common_runs
            # Move discarded interactive runs to discarded paths
            for run in discard_interactive:
                if session not in discarded_paths[key]:
                    discarded_paths[key][session] = {'interactive': {}, 'non_interactive': {}}
                discarded_paths[key][session]['interactive'][run] = paths_dict[key][session]['interactive'].pop(run)
                logger.info(f"Discarded interactive run {run} for session {session} in {key}.")
            # Move discarded non-interactive runs to discarded paths
            for run in discard_non_interactive:
                if session not in discarded_paths[key]:
                    discarded_paths[key][session] = {'interactive': {}, 'non_interactive': {}}
                discarded_paths[key][session]['non_interactive'][run] = paths_dict[key][session]['non_interactive'].pop(run)
                logger.info(f"Discarded non-interactive run {run} for session {session} in {key}.")
    # Update params with the pruned paths and the discarded paths
    params['data_file_paths'] = paths_dict
    params['discarded_paths'] = discarded_paths
    logger.info("Data file paths pruned successfully.")
    return params


def compute_or_load_variables(compute_func, load_func, file_paths, remake_flag_key, params, *args, **kwargs):
    """
    Generic method to manage compute vs. load actions for various data like gaze, fixations, saccades, etc.
    Parameters:
    - compute_func (function): The function that computes the data.
    - load_func (function): The function that loads the data from saved files.
    - file_paths (list): List of file paths where each variable will be saved or loaded from.
    - remake_flag_key (str): The key in params to check whether to compute or load (e.g., 'remake_gaze_data_dict').
    - params (dict): Dictionary of parameters, including the remake flag.
    - args, kwargs: Additional arguments to pass to the compute_func.
    Returns:
    - A list of variables, either loaded from files or computed.
    """
    remake_flag = params.get(remake_flag_key, True)  # Check the corresponding remake flag
    if remake_flag:
        logger.info(f"Remake flag '{remake_flag_key}' is set to True. Computing data using {compute_func.__name__}.")
        # Compute the data
        computed_vars = compute_func(*args, **kwargs)
        # Save each computed variable to its corresponding file path
        for file_path, var in zip(file_paths, computed_vars):
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(var, f)
                logger.info(f"Saved computed data to {file_path}.")
            except Exception as e:
                logger.error(f"Failed to save computed data to {file_path}: {e}")
        return computed_vars
    else:
        logger.info(f"Remake flag '{remake_flag_key}' is set to False. Loading data using {load_func.__name__}.")
        try:
            # Load the data using the provided load function
            loaded_vars = load_func(*file_paths)
            logger.info(f"Successfully loaded data from {file_paths}.")
            return loaded_vars
        except Exception as e:
            logger.error(f"Failed to load data from {file_paths}: {e}")
            raise


def prune_nans_in_specific_timeseries(time_series, positions, pupil_size):
    """
    Prunes NaN values from the time series and adjusts the corresponding position and pupil_size vectors.
    Ensures m1 and m2 data are synchronized with the time series, having no NaNs and the same number of points.
    Parameters:
    - time_series (np.ndarray): The time series array.
    - positions (dict): A dictionary containing position data with keys 'm1' and optionally 'm2'.
    - pupil_size (dict): A dictionary containing pupil size data with keys 'm1' and optionally 'm2'.
    Returns:
    - pruned_positions (dict): The positions dictionary with NaN values pruned.
    - pruned_pupil_size (dict): The pupil size dictionary with NaN values pruned.
    - pruned_time_series (np.ndarray): The pruned time series array.
    """
    # Find valid indices in the time series where there are no NaNs
    valid_time_indices = ~np.isnan(time_series).flatten()
    # Initialize valid indices to the time indices
    valid_indices = valid_time_indices
    # Check and combine valid indices from positions and pupil size for both m1 and m2
    for key in ['m1', 'm2']:
        # Check positions[key] if present, not None, and non-empty
        if key in positions and positions[key] is not None and positions[key].size > 0:
            # Ensure positions[key] is 2D, and prune based on valid indices
            valid_indices = valid_indices & ~np.isnan(positions[key]).any(axis=0)
        # Check pupil_size[key] if present, not None, and non-empty
        if key in pupil_size and pupil_size[key] is not None and pupil_size[key].size > 0:
            # Ensure pupil_size[key] is 1D, and prune based on valid indices
            valid_indices = valid_indices & ~np.isnan(pupil_size[key]).flatten()
    # Prune the time series based on combined valid indices
    pruned_time_series = time_series[valid_indices]
    # Prune positions for m1 and m2 based on the combined valid indices
    pruned_positions = {}
    for key in ['m1', 'm2']:
        if key in positions and positions[key] is not None and positions[key].size > 0:
            pruned_positions[key] = positions[key][:, valid_indices]
        else:
            pruned_positions[key] = np.array([])
    # Prune pupil size for m1 and m2 based on the combined valid indices
    pruned_pupil_size = {}
    for key in ['m1', 'm2']:
        if key in pupil_size and pupil_size[key] is not None and pupil_size[key].size > 0:
            pruned_pupil_size[key] = pupil_size[key][:, valid_indices]
        else:
            pruned_pupil_size[key] = np.array([])
    return pruned_positions, pruned_pupil_size, pruned_time_series


























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

