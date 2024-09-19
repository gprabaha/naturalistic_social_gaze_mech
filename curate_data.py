#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17::48:42 2024

@author: pg496
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import load_data


def get_gaze_data_dict(data_file_paths):
    """
    Loads position, time, and pupil size data from the specified paths into a structured dictionary
    with parallel processing for each run.
    Parameters:
    - data_file_paths (dict): A dictionary containing paths to position, time, and pupil size files
      categorized by session, interaction type, and run number.
    Returns:
    - gaze_data_dict (dict): Dictionary structured as {session: {run: {'positions': {'m1': m1, 'm2': m2}, 'time': t, 'pupil_size': {'m1': m1, 'm2': m2}}}}.
    """
    gaze_data_dict = {}
    temp_results = []  # Temporary storage for parallel results
    # Using ThreadPoolExecutor for parallel data loading
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each run in parallel
        futures = []
        for data_type, sessions in data_file_paths.items():
            if data_type == 'legend':  # Skip legend if present
                continue
            for session, interaction_types in sessions.items():
                for interaction_type, runs in interaction_types.items():
                    for run, file_path in runs.items():
                        futures.append(executor.submit(load_run_data, data_type, session, run, file_path))
        # Collect the results as they complete
        for future in as_completed(futures):
            temp_results.append(future.result())
    # Assign the results to the gaze_data_dict
    for session, run, data_key, data_value in temp_results:
        # Initialize nested structure in the dictionary if not present
        if session not in gaze_data_dict:
            gaze_data_dict[session] = {}
        if run not in gaze_data_dict[session]:
            gaze_data_dict[session][run] = {}
        # Assign the loaded data to the appropriate key
        gaze_data_dict[session][run][data_key] = data_value
    return gaze_data_dict

def load_run_data(data_type, session, run, file_path):
    """
    Loads data for a specific run based on the data type.
    Parameters:
    - data_type (str): The type of data ('positions', 'neural_timeline', 'pupil_size').
    - session (str): The session identifier (usually a date).
    - run (int): The run number.
    - file_path (str): The path to the .mat file.
    Returns:
    - tuple: (session, run, data_key, data_value) where data_key is the type of data ('positions', 'time', 'pupil_size')
      and data_value is the corresponding data loaded from the file.
    """
    if data_type == 'positions':
        m1_data, m2_data = process_position_file(file_path)
        return session, run, 'positions', {'m1': m1_data, 'm2': m2_data}
    elif data_type == 'neural_timeline':
        t_data = process_time_file(file_path)
        return session, run, 'time', t_data
    elif data_type == 'pupil_size':
        m1_data, m2_data = process_pupil_file(file_path)
        return session, run, 'pupil_size', {'m1': m1_data, 'm2': m2_data}
    return session, run, data_type, None


def process_position_file(mat_file):
    """
    Processes a position file to extract m1 and m2 data.
    Parameters:
    - mat_file (str): Path to the .mat file.
    Returns:
    - tuple: (m1 data, m2 data) extracted from the .mat file.
    """
    mat_data = load_data.load_mat_from_path(mat_file)
    if 'var' in mat_data:
        aligned_positions = mat_data['var'][0][0]
        if 'm1' in aligned_positions.dtype.names and 'm2' in aligned_positions.dtype.names:
            return tuple(aligned_positions['m1'].T), tuple(aligned_positions['m2'].T)
    return None, None


def process_time_file(mat_file):
    """
    Processes a time file to extract time data.
    Parameters:
    - mat_file (str): Path to the .mat file.
    Returns:
    - ndarray: Time data extracted from the .mat file.
    """
    mat_data = load_data.load_mat_from_path(mat_file)
    if 'var' in mat_data:
        t = mat_data['var'][0][0]['t']
        return t
    return None


def process_pupil_file(mat_file):
    """
    Processes a pupil size file to extract m1 and m2 pupil size data.
    Parameters:
    - mat_file (str): Path to the .mat file.
    Returns:
    - tuple: (m1 pupil data, m2 pupil data) extracted from the .mat file.
    """
    mat_data = load_data.load_mat_from_path(mat_file)
    if 'var' in mat_data:
        pupil_data = mat_data['var'][0][0]
        if 'm1' in pupil_data.dtype.names and 'm2' in pupil_data.dtype.names:
            return tuple(pupil_data['m1'].T), tuple(pupil_data['m2'].T)
    return None, None
