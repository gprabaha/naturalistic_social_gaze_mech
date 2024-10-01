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


def compute_or_load_variables(compute_func, load_func, file_paths, remake_flag_key, params, *args, **kwargs):
    """
    Generic method to manage compute vs. load actions for various data like gaze, fixations, saccades, etc.
    Parameters:
    - compute_func (function): The function that computes the data.
    - load_func (function): The function that loads the data from saved files.
    - file_paths (list or str): List of file paths where each variable will be saved or loaded from, or a single path.
    - remake_flag_key (str): The key in params to check whether to compute or load.
    - params (dict): The dictionary containing configuration parameters.
    - args, kwargs: Additional arguments to pass to the compute_func.
    Returns:
    - A list of variables, either loaded from files or computed, or a single variable if there's only one output.
    """
    remake_flag = params.get(remake_flag_key, True)  # Check the corresponding remake flag
    if remake_flag:
        logger.info(f"Remake flag '{remake_flag_key}' is set to True. Computing data using {compute_func.__name__}.")
        # Compute the data
        computed_vars = compute_func(*args, params=params, **kwargs)  # Pass params as a keyword argument
        # If the computed_vars is not a tuple or list, make it a single-element list for consistent handling
        is_single_output = not isinstance(computed_vars, (list, tuple))
        if is_single_output:
            computed_vars = [computed_vars]
            file_paths = [file_paths]
        # Save each computed variable to its corresponding file path
        for file_path, var in zip(file_paths, computed_vars):
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(var, f)
                logger.info(f"Saved computed data to {file_path}.")
            except Exception as e:
                logger.error(f"Failed to save computed data to {file_path}: {e}")
        return computed_vars[0] if is_single_output else computed_vars
    else:
        logger.info(f"Remake flag '{remake_flag_key}' is set to False. Loading data using {load_func.__name__}.")
        try:
            # Handle loading for both single and multiple paths
            if isinstance(file_paths, str):
                loaded_vars = load_func(file_paths)
            else:
                loaded_vars = load_func(*file_paths)
            logger.info(f"Successfully loaded data from {file_paths}.")
            return loaded_vars
        except Exception as e:
            logger.error(f"Failed to load data from {file_paths}: {e}")
            raise


def generate_behav_dict_legend(data_dict, max_examples=2):
    """
    Generates a concise legend describing the nested structure of the given data dictionary.
    Parameters:
    - data_dict (dict): The dictionary whose structure is to be described.
    - max_examples (int): Maximum number of example paths to include for illustration.
    Returns:
    - legend (dict): A dictionary explaining the high-level structure of the input dictionary.
    """
    legend = {}
    if not isinstance(data_dict, dict) or not data_dict:
        return {'error': 'Empty or invalid data structure provided.'}
    # Summarize the structure at each level with a few examples
    legend['description'] = (
        "This legend summarizes the structure of the data. The dictionary contains several levels, each representing "
        "different aspects of the data."
    )
    # Fetch example keys for each level to ensure they match the intended descriptions
    # Level 1 should be the root keys, i.e., session numbers
    level_1_examples = list(data_dict.keys())[:max_examples]  # Direct access to root keys
    level_2_examples = fetch_dict_keys_at_level(data_dict, max_examples, target_level=1)
    level_3_examples = fetch_dict_keys_at_level(data_dict, max_examples, target_level=2)
    level_4_examples = fetch_dict_keys_at_level(data_dict, max_examples, target_level=3)
    # Add descriptions for each level with properly fetched examples
    legend['levels'] = {
        'level_1': {
            'description': "Top-level keys are session dates (8 digits).",
            'examples': [key for key in level_1_examples if isinstance(key, str) and key.isdigit()]
        },
        'level_2': {
            'description': "Second-level keys represent interaction types (e.g., interactive, non_interactive).",
            'examples': [key for key in level_2_examples if key in ['interactive', 'non_interactive']]
        },
        'level_3': {
            'description': "Third-level keys represent runs (integer values).",
            'examples': [key for key in level_3_examples if isinstance(key, int)]
        },
        'level_4': {
            'description': "Fourth-level keys represent data types (positions, neural timeline, pupil size).",
            'examples': [key for key in level_4_examples if key in ['positions', 'neural_timeline', 'pupil_size', 'roi_rects']]
        },
        'level_5': {
            'description': "Fifth-level keys contain m1 and m2 data under positions and pupil size.",
            'examples': ['m1', 'm2']
        }
    }
    return legend


def fetch_dict_keys_at_level(current_dict, max_examples, current_level=0, target_level=0):
    """
    Recursively fetches keys at a specified level within a nested dictionary.
    Parameters:
    - current_dict (dict): The current level of the dictionary.
    - max_examples (int): Maximum number of keys to collect.
    - current_level (int): Current level of recursion.
    - target_level (int): The target level to fetch keys from.
    Returns:
    - list: A list of example keys at the specified level.
    """
    keys = []
    if current_level == target_level and isinstance(current_dict, dict):
        keys.extend(list(current_dict.keys())[:max_examples])
        return keys
    for key, value in current_dict.items():
        if isinstance(value, dict):
            keys.extend(fetch_dict_keys_at_level(value, max_examples, current_level + 1, target_level))
        if len(keys) >= max_examples:
            break
    return keys[:max_examples]


def check_non_interactive_data(gaze_data_dict):
    """
    Checks if any 'non_interactive' key in the gaze data dictionary has data and identifies which runs contain data.
    Parameters:
    - gaze_data_dict (dict): A dictionary structured with sessions as top-level keys, followed by 
      'interactive' and 'non_interactive' keys, which contain run numbers and their associated data.
    Returns:
    - results (dict): A dictionary summarizing the sessions and runs where 'non_interactive' data is found,
      including details on the data keys ('positions', 'pupil_size', 'neural_timeline') and subkeys ('m1', 'm2').
    """
    results = {}
    # Iterate over each session in the gaze data dictionary
    for session_name, interaction_types in gaze_data_dict.items():
        # Check if 'non_interactive' exists in the interaction types
        if 'non_interactive' in interaction_types:
            # Iterate over each run in the 'non_interactive' interaction type
            for run_number, run_data in interaction_types['non_interactive'].items():
                # Initialize a flag to track if this run has any data
                has_data = False
                run_results = {}
                # Check positions data for m1 and m2
                if 'positions' in run_data:
                    positions = run_data['positions']
                    m1_data = positions.get('m1')
                    m2_data = positions.get('m2')
                    m1_has_data = m1_data is not None and m1_data.size > 0
                    m2_has_data = m2_data is not None and m2_data.size > 0
                    if m1_has_data or m2_has_data:
                        run_results['positions'] = {
                            'm1': m1_has_data,
                            'm2': m2_has_data
                        }
                        has_data = True
                # Check pupil size data for m1 and m2
                if 'pupil_size' in run_data:
                    pupil_size = run_data['pupil_size']
                    m1_data = pupil_size.get('m1')
                    m2_data = pupil_size.get('m2')
                    m1_has_data = m1_data is not None and m1_data.size > 0
                    m2_has_data = m2_data is not None and m2_data.size > 0
                    if m1_has_data or m2_has_data:
                        run_results['pupil_size'] = {
                            'm1': m1_has_data,
                            'm2': m2_has_data
                        }
                        has_data = True
                # Check neural timeline data
                if 'neural_timeline' in run_data and run_data['neural_timeline'] is not None and run_data['neural_timeline'].size > 0:
                    run_results['neural_timeline'] = True
                    has_data = True
                # If any data was found, add it to the results
                if has_data:
                    if session_name not in results:
                        results[session_name] = {}
                    results[session_name][run_number] = run_results
    # Log and return the results
    if results:
        print("Found non_interactive data in the following sessions and runs:")
        for session, runs in results.items():
            for run, data_keys in runs.items():
                print(f"Session: {session}, Run: {run}, Data: {data_keys}")
    else:
        print("No non_interactive data found in the gaze data dictionary.")
    return results


def print_dict_keys(d, indent=0):
    """Recursively prints keys of a dictionary. If a value is a dictionary,
    it will print its keys as well. Halts if value is not a dictionary."""
    for key, value in d.items():
        print(' ' * indent + str(key))  # Print key with indentation
        if isinstance(value, dict):     # If the value is a dictionary, recurse
            print_dict_keys(value, indent + 2)




























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

