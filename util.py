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
        # Check if the compute function accepts params and pass it accordingly
        try:
            computed_vars = compute_func(*args, **kwargs)  # First try without params
        except Exception as e:
            logger.error(f"Failed to compute function {compute_func.__name__}: {e}")
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


def reshape_to_ensure_data_rows_represent_samples(array):
    """
    Ensures that the rows of the array represent samples, i.e., reshapes any axN matrix into Nxa,
    but keeps an Nxa input unchanged.
    Parameters:
    - array (np.ndarray): The array to reshape. Should be a 2D matrix.
    Returns:
    - reshaped_array (np.ndarray): The reshaped array, with rows representing samples (Nx1 or Nx2).
    """
    if array is None:
        return None
    # Check if it's a 2D array (axN) and reshape to Nxa if necessary
    if array.ndim == 2:
        if array.shape[0] < array.shape[1]:
            return array.T  # Transpose if it's axN to make it Nxa
        else:
            return array  # Return as is if it's already Nxa
    # If it's a 1D array, reshape to Nx1
    if array.ndim == 1:
        return array.reshape(-1, 1)
    # Return the array as-is for other cases (if any)
    return array


def initiate_behav_df_label_cols(gaze_data_df):
    # Select only the desired columns
    behav_df = gaze_data_df[['session_name', 'interaction_type', 'run_number', 'agent']]
    return behav_df


def verify_presence_of_recording_sessions(recording_sessions_df, gaze_data_df, session_column='session_name'):
    # Get sets of session names from both dataframes
    sessions_in_recording = set(recording_sessions_df[session_column].unique())
    sessions_in_gaze_data = set(gaze_data_df[session_column].unique()) 
    # Find sessions in recording_sessions_df but not in gaze_data_df
    missing_in_gaze_data = sessions_in_recording - sessions_in_gaze_data
    # Find sessions in gaze_data_df but not in recording_sessions_df
    extra_in_gaze_data = sessions_in_gaze_data - sessions_in_recording
    # Report missing and extra sessions
    if missing_in_gaze_data:
        print(f"Sessions in recording_sessions_df but missing in gaze_data_df: {missing_in_gaze_data}")
    else:
        print("All sessions in recording_sessions_df are present in gaze_data_df.")
    if extra_in_gaze_data:
        print(f"Sessions in gaze_data_df but not in recording_sessions_df: {extra_in_gaze_data}")
    else:
        print("No extra sessions in gaze_data_df.")
    return missing_in_gaze_data, extra_in_gaze_data










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


def print_dict_keys(d, indent=0, limit=2):
    """Recursively prints keys of a dictionary. Limits the number of keys printed at each level.
    If more than 'limit' keys exist at a level, it prints an ellipsis."""
    keys = list(d.keys())  # Get all the keys in the dictionary
    for i, key in enumerate(keys):
        if i >= limit:
            # Print a vertical ellipsis (one dot per line)
            print(' ' * indent + '.')
            print(' ' * indent + '.')
            print(' ' * indent + '.')
            break
        print(' ' * indent + str(key))  # Print the key with the specified indentation
        value = d[key]
        if isinstance(value, dict):     # If the value is a dictionary, recurse
            print_dict_keys(value, indent + 2, limit)


def print_dict_keys_and_values(d, indent=0, limit=2, value_preview_limit=5):
    """Recursively prints keys of a dictionary. Limits the number of keys printed at each level.
    If more than 'limit' keys exist at a level, it prints a vertical ellipsis.
    At the bottom level, prints part of the value if it's not a dictionary.
    Truncates long values (like lists or arrays) to a preview size."""
    keys = list(d.keys())  # Get all the keys in the dictionary
    for i, key in enumerate(keys):
        if i >= limit:
            # Print a vertical ellipsis (one dot per line)
            print(' ' * indent + '.')
            print(' ' * indent + '.')
            print(' ' * indent + '.')
            break
        value = d[key]
        print(' ' * indent + str(key))  # Print the key with indentation
        if isinstance(value, dict):     # If the value is a dictionary, recurse
            print_dict_keys_and_values(value, indent + 2, limit, value_preview_limit)
        else:
            # Print a preview of the value
            _preview_value(value, indent + 2, value_preview_limit)


def _preview_value(value, indent, preview_limit):
    """Helper function to print a preview of a value, truncating if it's large."""
    preview_indent = ' ' * indent
    if isinstance(value, (list, tuple)):
        if len(value) > preview_limit:
            print(f"{preview_indent}{value[:preview_limit]}... (and {len(value) - preview_limit} more)")
        else:
            print(f"{preview_indent}{value}")
    elif isinstance(value, (str, bytes)):
        if len(value) > preview_limit:
            print(f"{preview_indent}{value[:preview_limit]}... (and {len(value) - preview_limit} more characters)")
        else:
            print(f"{preview_indent}{value}")
    else:
        print(f"{preview_indent}{value}")  # For other types, just print the value


def merge_dictionaries(dest_dict, src_dict):
    """
    Recursively merges source dictionary into the destination dictionary, preserving all keys.
    Args:
        dest_dict: The destination dictionary to merge into.
        src_dict: The source dictionary whose values will be merged into the destination.
    """
    for key, value in src_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recursively merge it
            node = dest_dict.setdefault(key, {})
            merge_dictionaries(node, value)
        else:
            # Otherwise, directly set the value in the destination dictionary
            dest_dict[key] = value


