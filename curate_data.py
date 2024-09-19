#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17::48:42 2024

@author: pg496
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import load_data

import pdb


# Set up a logger for this script
logger = logging.getLogger(__name__)


def get_gaze_data_dict(data_file_paths, use_parallel):
    """
    Loads position, time, and pupil size data from the specified paths into a structured dictionary
    with optional parallel processing based on `use_parallel`.
    Parameters:
    - data_file_paths (dict): A dictionary containing paths to position, time, and pupil size files
      categorized by session, interaction type, and run number.
    - use_parallel (bool): If True, uses parallel processing; if False, processes sequentially.
    Returns:
    - gaze_data_dict (dict): Dictionary structured as {session: {interaction_type: {run: {'positions': {'m1': m1, 'm2': m2}, 
      'time': t, 'pupil_size': {'m1': m1, 'm2': m2}}}}}.
    """
    logger.info("Starting to load gaze data from specified paths.")
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
                            logger.debug(f"Submitting task for session: {session}, run: {run}, interaction: {interaction_type}, data type: {data_type}")
                            futures.append(executor.submit(load_run_data, data_type, session, interaction_type, run, file_path))
            # Collect the results as they complete with tqdm progress bar
            for future in tqdm(as_completed(futures), total=total_tasks, desc="Loading Data", unit="file"):
                try:
                    temp_results.append(future.result())
                    logger.debug("Successfully loaded data for a run.")
                except Exception as e:
                    logger.error(f"Error loading data: {e}")
    else:
        logger.info("Using serial processing to load data.")
        # Serial processing
        for data_type, sessions in data_file_paths.items():
            if data_type == 'legend':  # Skip legend if present
                continue
            for session, interaction_types in sessions.items():
                for interaction_type, runs in interaction_types.items():
                    for run, file_path in runs.items():
                        try:
                            logger.debug(f"Processing session: {session}, run: {run}, interaction: {interaction_type}, data type: {data_type}")
                            result = load_run_data(data_type, session, interaction_type, run, file_path)
                            temp_results.append(result)
                            logger.debug("Successfully loaded data for a run.")
                        except Exception as e:
                            logger.error(f"Error loading data for session: {session}, run: {run}, file: {file_path}. Error: {e}")
    # Assign the results to the gaze_data_dict
    for session, interaction_type, run, data_key, data_value in temp_results:
        if session not in gaze_data_dict:
            gaze_data_dict[session] = {}
        if interaction_type not in gaze_data_dict[session]:
            gaze_data_dict[session][interaction_type] = {}
        if run not in gaze_data_dict[session][interaction_type]:
            gaze_data_dict[session][interaction_type][run] = {}
        gaze_data_dict[session][interaction_type][run][data_key] = data_value
        logger.debug(f"Assigned data to gaze_data_dict for session: {session}, interaction: {interaction_type}, run: {run}, data key: {data_key}")
    logger.info("Completed loading gaze data.")
    return gaze_data_dict


def load_run_data(data_type, session, interaction_type, run, file_path):
    """
    Loads data for a specific run based on the data type and interaction type.
    Parameters:
    - data_type (str): The type of data ('positions', 'neural_timeline', 'pupil_size').
    - session (str): The session identifier (usually a date).
    - interaction_type (str): The type of interaction ('interactive' or 'non_interactive').
    - run (int): The run number.
    - file_path (str): The path to the .mat file.
    Returns:
    - tuple: (session, interaction_type, run, data_key, data_value) where data_key is the type of data ('positions', 'time', 'pupil_size')
      and data_value is the corresponding data loaded from the file.
    """
    try:
        if data_type == 'positions':
            m1_data, m2_data = process_position_file(file_path)
            return session, interaction_type, run, 'positions', {'m1': m1_data, 'm2': m2_data}
        elif data_type == 'neural_timeline':
            t_data = process_time_file(file_path)
            return session, interaction_type, run, 'time', t_data
        elif data_type == 'pupil_size':
            m1_data, m2_data = process_pupil_file(file_path)
            return session, interaction_type, run, 'pupil_size', {'m1': m1_data, 'm2': m2_data}
    except Exception as e:
        logger.error(f"Error loading data for session: {session}, interaction: {interaction_type}, run: {run}, file: {file_path}. Error: {e}")
    return session, interaction_type, run, data_type, None



def process_position_file(mat_file):
    """
    Processes a position file to extract m1 and m2 data.
    Parameters:
    - mat_file (str): Path to the .mat file.
    Returns:
    - tuple: (m1 data, m2 data) extracted from the .mat file.
    """
    mat_data = load_data.load_mat_from_path(mat_file)
    pdb.set_trace()
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
