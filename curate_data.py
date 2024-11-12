#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17::48:42 2024

@author: pg496
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from tqdm import tqdm
import os
import re
import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
import scipy

import load_data
import util

import pdb


# Set up a logger for this script
logger = logging.getLogger(__name__)


def add_root_data_to_params(params):
    """
    Sets the root data directory based on cluster and Grace settings.
    Parameters:
    - params (dict): Dictionary containing configuration parameters, including flags for cluster and Grace.
    Returns:
    - params (dict): Updated dictionary with the 'root_data_dir' field added.
    """
    logger.info("Setting root data directory based on whether on the cluster or not and whether on Grace or Milgram.")
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
    path_to_roi_rects = os.path.join(root_data_dir, 'eyetracking/roi_rects')
    params['positions_dir'] = path_to_positions
    params['neural_timeline_dir'] = path_to_time_vecs
    params['pupil_size_dir'] = path_to_pupil_vecs
    params['roi_rects_dir'] = path_to_roi_rects
    logger.info("Raw data directories added to params.")
    return params


def add_paths_to_all_data_files_to_params(params):
    """
    Populates the paths to data files categorized by session, interaction type, run number, and data type
    into a pandas DataFrame for easier traversal and querying.
    Parameters:
    - params (dict): Dictionary containing paths to 'positions_dir', 'neural_timeline_dir', 'pupil_size_dir', and 'roi_rects_dir'.
    Returns:
    - params (dict): Updated dictionary with 'data_file_paths_df' field, containing a DataFrame with categorized paths.
    """
    # Define directories
    directories = {
        'positions': params['positions_dir'],
        'neural_timeline': params['neural_timeline_dir'],
        'pupil_size': params['pupil_size_dir'],
        'roi_rects': params['roi_rects_dir']
    }
    # Dictionary to collect files by session, interaction type, and run number
    collected_paths = {}
    # Define regex to extract session name, file type, and run number
    file_pattern = re.compile(r'(\d{8})_(position|dot)_(\d+)\.mat')
    # Iterate over each directory
    for data_type, dir_path in directories.items():
        try:
            for filename in os.listdir(dir_path):
                match = file_pattern.match(filename)
                if match:
                    session_name, file_type, run_number = match.groups()
                    run_number = int(run_number)
                    # Determine interaction type based on file type
                    interaction_type = 'interactive' if file_type == 'position' else 'non_interactive'
                    # Initialize the key for this session, interaction type, and run
                    key = (session_name, interaction_type, run_number)
                    if key not in collected_paths:
                        collected_paths[key] = {
                            'positions': None,
                            'neural_timeline': None,
                            'pupil_size': None,
                            'roi_rects': None
                        }
                    # Assign the file path to the appropriate data type
                    if data_type == 'positions':
                        collected_paths[key]['positions'] = os.path.join(dir_path, filename)
                    elif data_type == 'neural_timeline':
                        collected_paths[key]['neural_timeline'] = os.path.join(dir_path, filename)
                    elif data_type == 'pupil_size':
                        collected_paths[key]['pupil_size'] = os.path.join(dir_path, filename)
                    elif data_type == 'roi_rects':
                        collected_paths[key]['roi_rects'] = os.path.join(dir_path, filename)
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")
            # Check and log missing values
    for key, paths in collected_paths.items():
        if any(value is None for value in paths.values()):  # Explicitly check for None
            logger.warning(f"Missing data for session {key[0]}, interaction {key[1]}, run {key[2]}: {paths}; will be pruned out of paths df")
    # Filter out entries that contain None values explicitly
    complete_paths_list = [
        {
            'session_name': key[0],
            'interaction_type': key[1],
            'run_number': key[2],
            **paths
        }
        for key, paths in collected_paths.items()
        if all(value is not None for value in paths.values())  # Explicitly check for None
    ]
    # Convert the list to a DataFrame
    paths_df = pd.DataFrame(complete_paths_list)
    # Update params with the paths DataFrame
    params['data_file_paths_df'] = paths_df
    return params


def prune_data_file_paths_with_pos_time_filename_mismatch(params):
    """
    Prunes the data file paths DataFrame to ensure that the filenames of positions, neural timeline, and pupil size 
    are consistent within each run (based on session, interaction type, and run number).
    """
    logger.info("Pruning data file paths to ensure consistency of filenames within each session and interaction type.")
    # Extract the paths DataFrame from params
    paths_df = params.get('data_file_paths_df', pd.DataFrame())
    if paths_df.empty:
        logger.warning("No data file paths found to prune.")
        return params
    # Group by session_name, interaction_type, and run_number
    grouped_df = paths_df.groupby(['session_name', 'interaction_type', 'run_number'])
    # Separate consistent and inconsistent runs by applying the consistency check within each group
    consistent_mask = grouped_df.apply(lambda group: _check_filenames_consistency_within_run(group.iloc[0]))
    # Reset index to match the original DataFrame's index
    consistent_mask = consistent_mask.reset_index(drop=True)
    # Filter for consistent and inconsistent runs
    consistent_paths_df = paths_df[consistent_mask]
    discarded_paths_df = paths_df[~consistent_mask]
    # Update params with the pruned paths and the discarded paths DataFrame
    params['data_file_paths_df'] = consistent_paths_df
    params['discarded_paths_df'] = discarded_paths_df
    # Log how many paths were discarded
    logger.info(f"Discarded {len(discarded_paths_df)} runs due to inconsistent or missing filenames within their respective session and interaction type.")
    return params


def _check_filenames_consistency_within_run(group):
    """
    Checks if the filenames of positions, neural timeline, pupil size, and roi_rects are consistent for a given run.
    Parameters:
    - group (pd.Series): A row from the DataFrame representing a single run with 'positions', 'neural_timeline', 
      'pupil_size', and 'roi_rects' fields.
    Returns:
    - bool: True if filenames are consistent and none are missing, otherwise False.
    """
    # Extract filenames from the full paths
    filenames = [
        os.path.basename(group['positions']) if group['positions'] else None,
        os.path.basename(group['neural_timeline']) if group['neural_timeline'] else None,
        os.path.basename(group['pupil_size']) if group['pupil_size'] else None,
        os.path.basename(group['roi_rects']) if group['roi_rects'] else None
    ]
    # Return False if any filename is missing (None)
    if any(filename is None for filename in filenames):
        return False
    # Return True if all filenames are the same, otherwise False
    return len(set(filenames)) == 1


def make_gaze_data_df(params):
    """
    Loads position, neural timeline, pupil size, and ROI rects data from the specified paths into a structured DataFrame
    with optional parallel processing based on `use_parallel`. Each agent ('m1' or 'm2') will have its own row 
    for positions, pupil size, and ROI rects, while neural timeline is the same for both agents.
    Parameters:
    - params (dict): A dictionary containing configuration parameters, including 'data_file_paths_df', 
      'use_parallel', and 'processed_data_dir'.
    Returns:
    - gaze_data_df (pd.DataFrame): DataFrame structured with columns ['session_name', 'interaction_type', 'run_number', 
      'agent', 'positions', 'neural_timeline', 'pupil_size', 'roi_rects'].
    - missing_data_info (list): List of dictionaries containing paths and information about which data is missing.
    """
    logger.info("Starting to load gaze data from specified paths.")
    data_file_paths_df = params.get('data_file_paths_df', pd.DataFrame())
    use_parallel = params.get('use_parallel', False)
    if data_file_paths_df.empty:
        logger.warning("No data file paths found.")
        return pd.DataFrame(), []
    temp_results = []  # Temporary storage for results
    missing_data_info = []  # To store information on missing or empty data
    total_tasks = len(data_file_paths_df)
    logger.info(f"Total tasks to process: {total_tasks}")
    if use_parallel:
        logger.info("Using parallel processing to load data.")
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_load_run_data, row['session_name'], row['interaction_type'], row['run_number'], row)
                for _, row in data_file_paths_df.iterrows()
            ]
            with tqdm(total=total_tasks, desc="Loading Data", unit="file", leave=False) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    temp_results.append(result)
                    pbar.update(1)
    else:
        logger.info("Using serial processing to load data.")
        with tqdm(total=total_tasks, desc="Loading Data", unit="file", leave=False) as pbar:
            for _, row in data_file_paths_df.iterrows():
                result = _load_run_data(row['session_name'], row['interaction_type'], row['run_number'], row)
                temp_results.append(result)
                pbar.update(1)
    pbar.close()
    # Process the results into a DataFrame
    rows = []
    for result in temp_results:
        session_name, interaction_type, run_number, \
            positions_m1, positions_m2, \
                neural_timeline, \
                    pupil_size_m1, pupil_size_m2, \
                        roi_rects_m1, roi_rects_m2 = result
        # Ensure positions, pupil sizes, and neural timeline are reshaped using the general reshaping function
        positions_m1 = util.reshape_to_ensure_data_rows_represent_samples(positions_m1)
        positions_m2 = util.reshape_to_ensure_data_rows_represent_samples(positions_m2)
        neural_timeline = util.reshape_to_ensure_data_rows_represent_samples(neural_timeline)
        pupil_size_m1 = util.reshape_to_ensure_data_rows_represent_samples(pupil_size_m1)
        pupil_size_m2 = util.reshape_to_ensure_data_rows_represent_samples(pupil_size_m2)
        # Add a row for agent 'm1'
        rows.append({
            'session_name': session_name,
            'interaction_type': interaction_type,
            'run_number': run_number,
            'agent': 'm1',
            'positions': positions_m1,
            'neural_timeline': neural_timeline,
            'pupil_size': pupil_size_m1,
            'roi_rects': roi_rects_m1
        })
        # Add a row for agent 'm2'
        rows.append({
            'session_name': session_name,
            'interaction_type': interaction_type,
            'run_number': run_number,
            'agent': 'm2',
            'positions': positions_m2,
            'neural_timeline': neural_timeline,  # Same neural timeline for both agents
            'pupil_size': pupil_size_m2,
            'roi_rects': roi_rects_m2
        })
    gaze_data_df = pd.DataFrame(rows)
    # Identify and log which data is missing for each session
    for idx, row in gaze_data_df.iterrows():
        missing_data = {}
        if row['positions'] is None or len(row['positions']) == 0:
            missing_data['positions'] = True
        if row['neural_timeline'] is None or len(row['neural_timeline']) == 0:
            missing_data['neural_timeline'] = True
        if row['pupil_size'] is None or len(row['pupil_size']) == 0:
            missing_data['pupil_size'] = True
        if row['roi_rects'] is None or not bool(row['roi_rects']):
            missing_data['roi_rects'] = True
        if missing_data:
            missing_data_info.append({
                'session_name': row['session_name'],
                'interaction_type': row['interaction_type'],
                'run_number': row['run_number'],
                'agent': row['agent']
            })
    if len(missing_data_info) > 0:
        logger.info(f"Found missing or empty data for {len(missing_data_info)} runs:\n{missing_data_info}")
    else:
        logger.info("No missing data found.")
    # Remove rows with any missing data
    gaze_data_df = gaze_data_df.dropna(subset=['positions', 'neural_timeline', 'pupil_size', 'roi_rects'])
    return gaze_data_df, missing_data_info



def _load_run_data(session, interaction_type, run, row):
    """
    Loads all relevant data for a given session, interaction type, and run, including positions, neural timeline,
    pupil size, and ROI rects.
    Parameters:
    - session (str): The session identifier.
    - interaction_type (str): The type of interaction ('interactive' or 'non_interactive').
    - run (int): The run number.
    - row (pd.Series): Row from the data paths DataFrame containing the paths for each data type.
    Returns:
    - tuple: A tuple containing session, interaction type, run, and the loaded data for all types.
    """
    positions_m1, positions_m2 = None, None
    neural_timeline = None
    pupil_size_m1, pupil_size_m2 = None, None
    roi_rects_m1, roi_rects_m2 = None, None
    # Load positions
    if row['positions']:
        positions_m1, positions_m2 = __process_position_file(row['positions'])
    # Load neural timeline
    if row['neural_timeline']:
        neural_timeline = __process_time_file(row['neural_timeline'])
    # Load pupil size
    if row['pupil_size']:
        pupil_size_m1, pupil_size_m2 = __process_pupil_file(row['pupil_size'])
    # Load ROI rects
    if row['roi_rects']:
        roi_rects_m1, roi_rects_m2 = __process_roi_rects_file(row['roi_rects'])
    return (
        session, interaction_type, run, 
        positions_m1, positions_m2, 
        neural_timeline, 
        pupil_size_m1, pupil_size_m2, 
        roi_rects_m1, roi_rects_m2
    )


def __process_position_file(mat_file):
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
        return m1_data, m2_data
    # Log and set a breakpoint if data is missing or improperly formatted
    logger.warning(f"Position data is missing or improperly formatted in file: {mat_file}")
    return None, None


def __process_time_file(mat_file):
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
    return time_data


def __process_pupil_file(mat_file):
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
        return m1_data, m2_data
    # Log and set a breakpoint if data is missing or improperly formatted
    logger.warning(f"Pupil size data is missing or improperly formatted in file: {mat_file}")
    return None, None


def __process_roi_rects_file(mat_file):
    """
    Processes an ROI rects file to extract m1 and m2 ROI data as dictionaries.
    Parameters:
    - mat_file (str): Path to the .mat file.
    Returns:
    - tuple: (m1_data, m2_data) where each is a dictionary with ROI names as keys
             and the rect coordinates as values. If m1 or m2 data is missing, their value will be None.
    """
    mat_data = load_data.load_mat_from_path(mat_file)
    roi_data = None
    # Check if 'roi_rects' exists in the loaded mat data
    if isinstance(mat_data, dict) and 'roi_rects' in mat_data:
        roi_data = mat_data['roi_rects'][0][0]
    if roi_data is not None:
        # Extract m1 and m2 data
        m1_data = ___extract_roi_dict(roi_data['m1']) if 'm1' in roi_data.dtype.names else None
        m2_data = ___extract_roi_dict(roi_data['m2']) if 'm2' in roi_data.dtype.names else None
        return m1_data, m2_data
    # Log if roi_data is missing or improperly formatted
    logger.warning(f"ROI rects data is missing or improperly formatted in file: {mat_file}")
    return None, None


def ___extract_roi_dict(agent_roi_data):
    """
    Extracts ROI data from the given agent data.
    Parameters:
    - agent_roi_data: The agent's ROI data.
    Returns:
    - dict: A dictionary with ROI names as keys and rect coordinates as values, or None if the data is missing.
    """
    if agent_roi_data is None:
        return None
    roi_dict = {}
    for roi_name in agent_roi_data.dtype.names:
        roi_dict[roi_name] = agent_roi_data[roi_name][0][0][0]  # Adjust based on the data structure
    return roi_dict


def make_spike_times_df(params):
    processed_data_dir = params['processed_data_dir']
    spike_filename = 'unit_spiketimes.mat'
    spike_filepath = os.path.join(processed_data_dir, spike_filename)
    # Load the .mat file
    mat_data = load_data.load_mat_from_path(spike_filepath)
    # Access the data within the loaded .mat file
    spike_data = mat_data['unit_spiketimes'][0][0]
    # Convert the structured array to a dictionary
    spiketimes_dict = {}
    for name in spike_data.dtype.names:
        spiketimes_dict[name] = _flatten_nested_arrays(spike_data[name].squeeze())
    # Create a DataFrame from the dictionary
    spiketimes_df = pd.DataFrame(spiketimes_dict)
    spiketimes_df.rename(columns={'session':'session_name'}, inplace=True)
    return spiketimes_df


def _flatten_nested_arrays(arr):
    """
    Flatten nested arrays inside the given array, handling multi-element arrays.
    """
    # Check if the array is a numpy object array
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # Iterate through the array
        return [
            elem.item() if isinstance(elem, np.ndarray) and elem.size == 1 else elem[0] if isinstance(elem, np.ndarray) and elem.size > 1 and isinstance(elem[0], np.ndarray) else elem
            for elem in arr
        ]
    return arr  # Return as-is if not object array


def prune_nan_values_in_timeseries(gaze_data_df):
    """
    Prunes NaN values from the time series in the gaze data DataFrame and ensures synchronized timelines across agents.
    Parameters:
    - gaze_data_df (pd.DataFrame): The gaze data DataFrame containing session, interaction type, run, and agent data.
    Returns:
    - pruned_gaze_data_df (pd.DataFrame): The pruned gaze data DataFrame with NaN values removed and synchronized across agents.
    """
    logger.info("Pruning NaN values and synchronizing timelines across agents in gaze data.")
    pruned_rows = []
    # Group by session_name, interaction_type, and run_number to handle each pair of agents together
    grouped = gaze_data_df.groupby(['session_name', 'interaction_type', 'run_number'])
    for _, group in grouped:
        # Separate data for each agent
        m1_data = group[group['agent'] == 'm1']
        m2_data = group[group['agent'] == 'm2']
        if not m1_data.empty and not m2_data.empty:
            m1_row = m1_data.iloc[0]
            m2_row = m2_data.iloc[0]
            # Prune NaN values and ensure synchronized pruning across both agents
            pruned_m1_positions, pruned_m1_pupil_size, pruned_m1_time_series, pruned_m2_positions, pruned_m2_pupil_size, pruned_m2_time_series = \
            _prune_nans_to_synchronize_agent_timelines(
                m1_row['neural_timeline'], m1_row['positions'], m1_row['pupil_size'],
                m2_row['neural_timeline'], m2_row['positions'], m2_row['pupil_size']
            )
            # Create new rows with pruned data
            pruned_m1_row = m1_row.copy()
            pruned_m1_row['positions'], pruned_m1_row['pupil_size'], pruned_m1_row['neural_timeline'] = pruned_m1_positions, pruned_m1_pupil_size, pruned_m1_time_series
            pruned_rows.append(pruned_m1_row)
            pruned_m2_row = m2_row.copy()
            pruned_m2_row['positions'], pruned_m2_row['pupil_size'], pruned_m2_row['neural_timeline'] = pruned_m2_positions, pruned_m2_pupil_size, pruned_m2_time_series
            pruned_rows.append(pruned_m2_row)
    # Convert the list of pruned rows into a DataFrame
    pruned_gaze_data_df = pd.DataFrame(pruned_rows)
    return pruned_gaze_data_df


def _prune_nans_to_synchronize_agent_timelines(m1_time_series, m1_positions, m1_pupil_size, m2_time_series, m2_positions, m2_pupil_size):
    """
    Prunes NaN values from both agents' time series and synchronizes the indices across both.
    Parameters:
    - m1_time_series, m1_positions, m1_pupil_size: Data arrays for agent m1.
    - m2_time_series, m2_positions, m2_pupil_size: Data arrays for agent m2.
    Returns:
    - Synchronized, pruned versions of positions, pupil sizes, and time series for both agents.
    """
    # Find valid indices for each agent where there are no NaNs
    m1_valid_indices = ~np.isnan(m1_time_series.flatten()) & ~np.isnan(m1_positions).any(axis=1) & ~np.isnan(m1_pupil_size.flatten())
    m2_valid_indices = ~np.isnan(m2_time_series.flatten()) & ~np.isnan(m2_positions).any(axis=1) & ~np.isnan(m2_pupil_size.flatten())
    util.print_nan_cluster_histogram(np.isnan(m1_time_series.flatten()))
    util.print_nan_cluster_histogram(np.isnan(m1_time_series.flatten()))
    pdb.set_trace()
    util.print_nan_cluster_histogram(np.isnan(m1_positions).any(axis=1))
    util.print_nan_cluster_histogram(np.isnan(m2_positions).any(axis=1))
    # Combine valid indices to keep only synchronized non-NaN entries
    synchronized_valid_indices = m1_valid_indices & m2_valid_indices
    pdb.set_trace()
    # Prune both agents' data using the synchronized indices
    pruned_m1_time_series = m1_time_series[synchronized_valid_indices]
    pruned_m1_positions = m1_positions[synchronized_valid_indices, :]
    pruned_m1_pupil_size = m1_pupil_size[synchronized_valid_indices]
    pruned_m2_time_series = m2_time_series[synchronized_valid_indices]
    pruned_m2_positions = m2_positions[synchronized_valid_indices, :]
    pruned_m2_pupil_size = m2_pupil_size[synchronized_valid_indices]
    return pruned_m1_positions, pruned_m1_pupil_size, pruned_m1_time_series, pruned_m2_positions, pruned_m2_pupil_size, pruned_m2_time_series



# Main function for adding fixation locations with serial or parallel processing and tqdm
def add_fixation_rois_in_dataframe(fixation_df, gaze_data_df, num_cpus, use_parallel=True):
    if use_parallel:
        with Pool(num_cpus) as pool:
            # Use tqdm with imap_unordered for real-time progress updates
            results = list(tqdm(
                pool.imap_unordered(
                    _process_fixation_row_wrapper, 
                    [(index, row, gaze_data_df) for index, row in fixation_df.iterrows()],
                    chunksize=num_cpus
                ),
                total=len(fixation_df),
                desc="Processing Fixation ROIs"
            ))
    else:
        # Serial processing with tqdm
        results = [
            (index, _process_fixation_row(row, gaze_data_df)) 
            for index, row in tqdm(fixation_df.iterrows(), total=len(fixation_df), desc="Processing Fixation ROIs")
        ]
    # Sort results to maintain original order
    results.sort(key=lambda x: x[0])
    ordered_location_labels = [result[1] for result in results]
    # Assign sorted results to the 'location' column of the dataframe
    fixation_df['location'] = ordered_location_labels
    return fixation_df


# Wrapper function for _process_fixation_row to handle multiprocessing arguments
def _process_fixation_row_wrapper(args):
    index, row, gaze_data_df = args
    return index, _process_fixation_row(row, gaze_data_df)


# Helper function for processing each row in the fixation dataframe
def _process_fixation_row(row, gaze_data_df):
    session_name = row['session_name']
    interaction_type = row['interaction_type']
    run_number = row['run_number']
    agent = row['agent']
    fixations = row['fixation_start_stop']
    
    # Filter the gaze data to match the current row's session, interaction type, run, and agent
    gaze_row = gaze_data_df[
        (gaze_data_df['session_name'] == session_name) &
        (gaze_data_df['interaction_type'] == interaction_type) &
        (gaze_data_df['run_number'] == run_number) &
        (gaze_data_df['agent'] == agent)
    ].iloc[0]
    positions = gaze_row['positions']
    roi_rects = gaze_row['roi_rects']
    location_labels = []
    # Process each fixation to identify all 'location' ROIs
    for start_stop in fixations:
        start_idx, stop_idx = start_stop
        fixation_positions = positions[start_idx:stop_idx + 1]
        mean_position = [
            sum(pos[0] for pos in fixation_positions) / len(fixation_positions),
            sum(pos[1] for pos in fixation_positions) / len(fixation_positions)
        ]
        # Gather all ROIs containing the mean position
        matched_rois = []
        for roi_name, rect in roi_rects.items():
            if rect[0] <= mean_position[0] <= rect[2] and rect[1] <= mean_position[1] <= rect[3]:
                matched_rois.append(roi_name)
        # Store all matched ROIs or 'elsewhere' if none matched
        location = matched_rois if matched_rois else ['elsewhere']
        location_labels.append(location)
    return location_labels


# Main function for adding saccade ROIs with serial or parallel processing and tqdm
def add_saccade_rois_in_dataframe(saccade_df, gaze_data_df, num_cpus, use_parallel=True):
    if use_parallel:
        with Pool(num_cpus) as pool:
            # Use tqdm with imap_unordered for real-time progress updates
            results = list(tqdm(
                pool.imap_unordered(
                    _process_saccade_row_wrapper, 
                    [(index, row, gaze_data_df) for index, row in saccade_df.iterrows()],
                    chunksize=num_cpus
                ),
                total=len(saccade_df),
                desc="Processing Saccade ROIs"
            ))
    else:
        # Serial processing with tqdm
        results = [
            (index, _process_saccade_row(row, gaze_data_df)) 
            for index, row in tqdm(saccade_df.iterrows(), total=len(saccade_df), desc="Processing Saccade ROIs")
        ]
    # Sort results to maintain original order
    results.sort(key=lambda x: x[0])
    ordered_from_labels = [result[1][0] for result in results]
    ordered_to_labels = [result[1][1] for result in results]
    # Assign sorted results to 'from' and 'to' columns of the dataframe
    saccade_df['from'] = ordered_from_labels
    saccade_df['to'] = ordered_to_labels
    return saccade_df


# Wrapper function for _process_saccade_row to handle multiprocessing arguments
def _process_saccade_row_wrapper(args):
    index, row, gaze_data_df = args
    return index, _process_saccade_row(row, gaze_data_df)


# Helper function for processing each row in the saccade dataframe
def _process_saccade_row(row, gaze_data_df):
    session_name = row['session_name']
    interaction_type = row['interaction_type']
    run_number = row['run_number']
    agent = row['agent']
    saccades = row['saccade_start_stop']
    
    # Filter the gaze data to match the current row's session, interaction type, run, and agent
    gaze_row = gaze_data_df[
        (gaze_data_df['session_name'] == session_name) &
        (gaze_data_df['interaction_type'] == interaction_type) &
        (gaze_data_df['run_number'] == run_number) &
        (gaze_data_df['agent'] == agent)
    ].iloc[0]
    positions = gaze_row['positions']
    roi_rects = gaze_row['roi_rects']
    from_labels, to_labels = [], []
    
    # Process each saccade to identify all 'from' and 'to' ROIs
    for start_stop in saccades:
        start_idx, stop_idx = start_stop
        start_pos = positions[start_idx]
        stop_pos = positions[stop_idx]
        
        # Gather all ROIs for start and stop positions
        from_rois = [roi_name for roi_name, rect in roi_rects.items() if rect[0] <= start_pos[0] <= rect[2] and rect[1] <= start_pos[1] <= rect[3]]
        to_rois = [roi_name for roi_name, rect in roi_rects.items() if rect[0] <= stop_pos[0] <= rect[2] and rect[1] <= stop_pos[1] <= rect[3]]
        
        # Assign 'elsewhere' if no ROIs match
        from_labels.append(from_rois if from_rois else ['elsewhere'])
        to_labels.append(to_rois if to_rois else ['elsewhere'])
    
    return from_labels, to_labels











def generate_binary_behav_timeseries_dicts(fixation_dict, saccade_dict):
    """
    Main function to generate binary vectors for fixation and saccade behaviors from the given dictionaries.
    Args:
        fixation_dict: Dictionary containing fixation data.
        saccade_dict: Dictionary containing saccade data.
    Returns:
        behavioral_vectors_dict: A dictionary with binary vectors for fixations and saccades.
    """
    behavioral_vectors_dict = {}
    # Process fixation and saccade dictionaries separately
    fixation_vectors = _generate_vectors_from_behavior_dict(fixation_dict, 'fixation')
    saccade_vectors = _generate_vectors_from_behavior_dict(saccade_dict, 'saccade')
    # Recursively merge fixation and saccade vectors
    util.merge_dictionaries(behavioral_vectors_dict, fixation_vectors)
    util.merge_dictionaries(behavioral_vectors_dict, saccade_vectors)
    return behavioral_vectors_dict


def _generate_vectors_from_behavior_dict(behavior_dict, behavior_type):
    """
    Generates binary vectors from the given behavior dictionary (fixations, saccades, etc.).
    Args:
        behavior_dict: Dictionary containing behavior data (fixation, saccade).
        behavior_type: The type of behavior (e.g., 'fixation', 'saccade').
    Returns:
        A dictionary with binary vectors for the specified behavior type.
    """
    behavioral_vectors = {}
    for session, session_data in behavior_dict.items():
        for interaction_type, interaction_data in session_data.items():
            for run_number, run_data in interaction_data.items():
                for agent, agent_data in run_data.items():
                    # Fetch XY and behavior indices
                    XY = agent_data['XY']
                    behavior_indices = agent_data[f'{behavior_type}indices']
                    # Generate the binary vector for the behavior
                    binary_vector = _generate_binary_vector_from_indices(XY, behavior_indices)
                    # Store the binary vector in the hierarchical dictionary using setdefault
                    behavioral_vectors.setdefault(session, {}).setdefault(interaction_type, {}).setdefault(run_number, {}).setdefault(agent, {})[behavior_type] = binary_vector
    return behavioral_vectors


def _generate_binary_vector_from_indices(XY, indices):
    """
    Helper function to generate a binary vector given the XY data and behavior indices.
    Args:
        XY: The 2xN or Nx2 position data timeseries array.
        indices: The start and stop indices for the behavior.
    Returns:
        A binary vector with 1s marking the behavior intervals and 0s elsewhere.
    """
    # Check the shape of indices to ensure it is always 2xN
    if indices.shape[0] != 2:
        indices = indices.T
    # Initialize binary vector of zeros with the length of XY
    data_length = XY.shape[1] if XY.shape[0] == 2 else XY.shape[0]
    binary_vector = np.zeros(data_length, dtype=int)
    # Efficiently set the values to 1 for each start-stop index pair
    for start, stop in indices.T:
        binary_vector[start:stop] = 1
    return binary_vector


from scipy.ndimage import gaussian_filter1d

def make_binned_unit_fr_df_for_each_run(spike_times_df, nan_removed_gaze_data_df, params):
    """
    Generate a binned firing rate DataFrame for each unit over each run, session, interaction type, and region.
    Args:
        spike_times_df (pd.DataFrame): DataFrame containing spike times of neurons.
        nan_removed_gaze_data_df (pd.DataFrame): DataFrame with gaze and session metadata.
        params (dict): Parameters dictionary containing 'neural_data_bin_size', 'downsample_bin_size', 'smoothing_sigma',
                       and 'use_parallel' (optional).
    Returns:
        pd.DataFrame: A DataFrame containing binned firing rate timeseries for each session, interaction type,
                      run, region, and unit.
    """
    # Extract parameters from params dictionary
    bin_size = params['neural_data_bin_size']
    downsample_bin_size = params['downsample_bin_size']  # 10 ms in seconds (0.01)
    sigma = params.get('smoothing_sigma', 5)
    use_parallel = params.get('use_parallel', False)
    # Get unique combinations of session, interaction type, and run
    run_groups = nan_removed_gaze_data_df.groupby(['session_name', 'interaction_type', 'run_number'])
    run_data = [(session, interaction, run, gaze_row) for (session, interaction, run), gaze_row in run_groups]
    # Choose execution strategy based on `use_parallel` flag
    if use_parallel:
        # Parallel execution
        fr_data = Parallel(n_jobs=params['num_cpus'])(
            delayed(_make_binned_neural_fr_timeseries_for_run)(
                session, interaction, run, gaze_row, spike_times_df, bin_size, downsample_bin_size, sigma
            ) for session, interaction, run, gaze_row in tqdm(run_data, desc="Processing in parallel")
        )
        fr_data = [item for sublist in fr_data for item in sublist]
    else:
        # Serial execution
        fr_data = []
        for session, interaction, run, gaze_row in tqdm(run_data, desc="Processing in serial"):
            fr_data.extend(
                _make_binned_neural_fr_timeseries_for_run(
                    session, interaction, run, gaze_row, spike_times_df, bin_size, downsample_bin_size, sigma
                )
            )
    # Convert list of records to a DataFrame
    fr_df = pd.DataFrame(fr_data)
    return fr_df


def _make_binned_neural_fr_timeseries_for_run(session, interaction, run, gaze_row, spike_times_df, bin_size, downsample_bin_size, sigma):
    """
    Helper function to create a binned neural firing rate timeseries for a specific session, interaction type, and run.
    Args:
        session (str): The session name.
        interaction (str): The interaction type.
        run (int): The run number.
        gaze_row (pd.DataFrame): A DataFrame row containing neural timeline data for the current run.
        spike_times_df (pd.DataFrame): DataFrame with spike times and neuron metadata.
        bin_size (float): The initial bin size (fine resolution) in seconds.
        downsample_bin_size (float): The final bin size in seconds (e.g., 0.01 for 10 ms).
        sigma (float): Sigma value for Gaussian smoothing.
    Returns:
        list: A list of dictionaries containing binned firing rate information for each unit and region.
    """
    # Get neural timeline start and end for this run
    neural_start = gaze_row['neural_timeline'].values[0][0].item()
    neural_end = gaze_row['neural_timeline'].values[0][-1].item()
    # Filter spike data for the current session
    session_spike_data = spike_times_df[spike_times_df['session_name'] == session]
    fr_data = []
    # Loop over each unique region and unit
    for (region, unit_uuid), unit_spikes in session_spike_data.groupby(['region', 'unit_uuid']):
        spike_times = np.concatenate(unit_spikes['spike_ts'].tolist())
        spike_times_in_run = spike_times[(spike_times >= neural_start) & (spike_times <= neural_end)]
        # Define fine bins and calculate firing rate in fine resolution
        num_fine_bins = int((neural_end - neural_start) / bin_size)
        # Histogram binning
        binned_counts, _ = np.histogram(spike_times_in_run, bins=num_fine_bins, range=(neural_start, neural_end))
        binned_counts = binned_counts.astype(float)
        smoothed_binned_counts = gaussian_filter1d(binned_counts, sigma=sigma)
        # Downsample by averaging over the desired 10 ms bins
        fine_to_final_ratio = int(downsample_bin_size / bin_size)
        num_final_bins = int(num_fine_bins / fine_to_final_ratio)
        downsampled_counts = smoothed_binned_counts[:num_final_bins * fine_to_final_ratio].reshape(-1, fine_to_final_ratio).mean(axis=1)
        # Convert counts to firing rate
        binned_fr = downsampled_counts / downsample_bin_size
        # Store results in a dictionary for each unit
        fr_data.append({
            'session_name': session,
            'interaction_type': interaction,
            'run_number': run,
            'region': region,
            'unit_uuid': unit_uuid,
            'binned_neural_fr_in_run': binned_fr
        })
    return fr_data


