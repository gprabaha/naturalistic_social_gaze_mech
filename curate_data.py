#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17::48:42 2024

@author: pg496
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import pickle

import load_data
import util

import pdb


# Set up a logger for this script
logger = logging.getLogger(__name__)


def make_gaze_data_dict(params):
    """
    Loads position, neural timeline, and pupil size data from the specified paths into a structured dictionary
    with optional parallel processing based on `use_parallel`. Saves the resulting dictionary and 
    missing data paths as separate pickle files.
    Parameters:
    - params (dict): A dictionary containing configuration parameters, including 'data_file_paths', 
      'use_parallel', and 'processed_data_dir'.
    Returns:
    - gaze_data_dict (dict): Dictionary structured as {session: {interaction_type: {run: {'positions': {'m1': m1_data, 'm2': m2_data}, 
      'neural_timeline': neural_timeline_data, 'pupil_size': {'m1': m1_data, 'm2': m2_data}}}}}.
    - missing_data_paths (list): List of paths where data is missing or empty.
    """
    logger.info("Starting to load gaze data from specified paths.")
    data_file_paths = params['data_file_paths']
    use_parallel = params.get('use_parallel', False)
    processed_data_dir = params['processed_data_dir']
    gaze_data_dict = {}
    temp_results = []  # Temporary storage for results
    # Prepare a list of tasks for tqdm progress bar
    total_tasks = sum(
        len(runs) 
        for data_type, sessions in data_file_paths.items() 
        if data_type != 'legend' 
        for interaction_types in sessions.values() 
        for runs in interaction_types.values()
    )
    logger.info(f"Total tasks to process: {total_tasks}")
    # Choose between parallel and serial processing
    if use_parallel:
        logger.info("Using parallel processing to load data.")
        with ThreadPoolExecutor() as executor:
            futures = []
            for data_type, sessions in data_file_paths.items():
                if data_type == 'legend':  # Skip legend if present
                    continue
                for session, interaction_types in sessions.items():
                    for interaction_type, runs in interaction_types.items():
                        for run, file_path in runs.items():
                            futures.append(executor.submit(load_run_data, data_type, session, interaction_type, run, file_path))
            # Collect the results as they complete with tqdm progress bar
            for future in tqdm(as_completed(futures), total=total_tasks, desc="Loading Data", unit="file"):
                temp_results.append(future.result())
    else:
        logger.info("Using serial processing to load data.")
        # Serial processing with a tqdm progress bar
        with tqdm(total=total_tasks, desc="Loading Data", unit="file") as pbar:
            for data_type, sessions in data_file_paths.items():
                if data_type == 'legend':  # Skip legend if present
                    continue
                for session, interaction_types in sessions.items():
                    for interaction_type, runs in interaction_types.items():
                        for run, file_path in runs.items():
                            result = load_run_data(data_type, session, interaction_type, run, file_path)
                            temp_results.append(result)
                            pbar.update(1)  # Manually update the progress bar
    # Assign the results to the gaze_data_dict
    for session, interaction_type, run, data_key, data_value in temp_results:
        session_dict = gaze_data_dict.setdefault(session, {})
        interaction_dict = session_dict.setdefault(interaction_type, {})
        run_dict = interaction_dict.setdefault(run, {})
        if data_key in ['positions', 'pupil_size']:
            # Handle m1 and m2 under positions and pupil size
            data_subdict = run_dict.setdefault(data_key, {})
            if 'm1' in data_value:
                data_subdict['m1'] = data_value['m1']
            if 'm2' in data_value:
                data_subdict['m2'] = data_value['m2']
        elif data_key == 'neural_timeline':
            # Directly assign neural_timeline data
            run_dict['neural_timeline'] = data_value
    logger.info("Completed loading gaze data.")
    # Add a concise legend to the gaze_data_dict
    gaze_data_dict['legend'] = util.generate_legend(gaze_data_dict)
    # Check the final structure and report any missing data
    missing_data_dict_paths, total_data_paths = check_dict_leaves(gaze_data_dict)
    if missing_data_dict_paths:
        logger.warning(f"Missing or empty data found at {len(missing_data_dict_paths)} data dict paths out of {total_data_paths} total paths.")
    else:
        logger.info(f"All data leaves are correctly populated. Total paths checked: {total_data_paths}.")
    return gaze_data_dict, missing_data_dict_paths




def load_run_data(data_type, session, interaction_type, run, file_path):
    if data_type == 'positions':
        m1_data, m2_data = process_position_file(file_path)
        return session, interaction_type, run, 'positions', {'m1': m1_data, 'm2': m2_data}
    elif data_type == 'neural_timeline':
        t_data = process_time_file(file_path)
        return session, interaction_type, run, 'time', t_data
    elif data_type == 'pupil_size':
        m1_data, m2_data = process_pupil_file(file_path)
        return session, interaction_type, run, 'pupil_size', {'m1': m1_data, 'm2': m2_data}
    return session, interaction_type, run, data_type, None


def process_position_file(mat_file):
    """
    Processes a position file to extract m1 and m2 data.
    Parameters:
    - mat_file (str): Path to the .mat file.
    Returns:
    - tuple: (m1 data, m2 data) extracted from the .mat file. If m1 or m2 is missing, it will return None for the missing data.
    """
    mat_data = load_data.load_mat_from_path(mat_file)
    # Explicitly check if 'var' or 'aligned_position_file' keys exist in the dictionary
    aligned_positions = None
    if isinstance(mat_data, dict) and 'var' in mat_data:
        aligned_positions = mat_data['var'][0][0]
    elif isinstance(mat_data, dict) and 'aligned_position_file' in mat_data:
        aligned_positions = mat_data['aligned_position_file'][0][0]
    if aligned_positions is not None:
        m1_data = aligned_positions['m1'] if 'm1' in aligned_positions.dtype.names else None
        m2_data = aligned_positions['m2'] if 'm2' in aligned_positions.dtype.names else None
        # Check if both data are missing or empty and set a breakpoint if it is
        if (m1_data is None or m1_data.size == 0) and (m2_data is None or m2_data.size == 0):
            logger.error(f"Both m1 and m2 data are missing or empty in file: {mat_file}")
            pdb.set_trace()  # Breakpoint will trigger if both m1 and m2 data are empty
        return m1_data, m2_data
    # Log and set a breakpoint if data is missing or improperly formatted
    logger.warning(f"Position data is missing or improperly formatted in file: {mat_file}")
    pdb.set_trace()  # Breakpoint when data is missing or improperly formatted
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
    # Explicitly check if 'var' or 'aligned_position_file' keys exist in the dictionary
    time_data = None
    if isinstance(mat_data, dict) and 'time_file' in mat_data:
        if 't' in mat_data['time_file'][0][0].dtype.names:
            time_data = mat_data['time_file'][0][0]['t']
    elif isinstance(mat_data, dict) and 'aligned_position_file' in mat_data:
        if 't' in mat_data['aligned_position_file'][0][0].dtype.names:
            time_data = mat_data['aligned_position_file'][0][0]['t']
    elif isinstance(mat_data, dict) and 'var' in mat_data:
        if 't' in mat_data['var'][0][0].dtype.names:
            time_data = mat_data['var'][0][0]['t']
    # Check if time data is empty or None and set a breakpoint if it is
    if time_data is None or time_data.size == 0:
        logger.error(f"Empty or missing time data in file: {mat_file}")
        pdb.set_trace()  # Breakpoint when time data is missing or improperly formatted
    return time_data


def process_pupil_file(mat_file):
    """
    Processes a pupil size file to extract m1 and m2 pupil size data.
    Parameters:
    - mat_file (str): Path to the .mat file.
    Returns:
    - tuple: (m1 pupil data, m2 pupil data) extracted from the .mat file. If m1 or m2 is missing, it will return None for the missing data.
    """
    mat_data = load_data.load_mat_from_path(mat_file)
    # Explicitly check if 'var' or 'aligned_position_file' keys exist in the dictionary
    pupil_data = None
    if isinstance(mat_data, dict) and 'var' in mat_data:
        pupil_data = mat_data['var'][0][0]
    elif isinstance(mat_data, dict) and 'aligned_position_file' in mat_data:
        pupil_data = mat_data['aligned_position_file'][0][0]
    if pupil_data is not None:
        m1_data = pupil_data['m1'] if 'm1' in pupil_data.dtype.names else None
        m2_data = pupil_data['m2'] if 'm2' in pupil_data.dtype.names else None
        # Check if both data are missing or empty and set a breakpoint if it is
        if (m1_data is None or m1_data.size == 0) and (m2_data is None or m2_data.size == 0):
            logger.error(f"Both m1 and m2 pupil data are missing or empty in file: {mat_file}")
            pdb.set_trace()  # Breakpoint will trigger if both m1 and m2 data are empty
        return m1_data, m2_data
    # Log and set a breakpoint if data is missing or improperly formatted
    logger.warning(f"Pupil size data is missing or improperly formatted in file: {mat_file}")
    pdb.set_trace()  # Breakpoint when data is missing or improperly formatted
    return None, None


def check_dict_leaves(data_dict):
    """
    Checks all leaves of a nested dictionary to find any missing or empty data.
    Parameters:
    - data_dict (dict): The dictionary to check.
    Returns:
    - missing_paths (list): List of paths where data is missing or empty.
    - total_paths (int): The total number of paths checked.
    """
    missing_paths = []
    total_paths = 0

    def recursive_check(d, path=""):
        nonlocal total_paths
        for key, value in d.items():
            current_path = f"{path}/{key}" if path else str(key)
            if isinstance(value, dict):
                recursive_check(value, current_path)
            else:
                total_paths += 1
                if value is None or (hasattr(value, "size") and value.size == 0):
                    missing_paths.append(current_path)

    recursive_check(data_dict)
    return missing_paths, total_paths
