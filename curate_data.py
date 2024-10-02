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
import re
import numpy as np
import pickle

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
    path_to_roi_rects = os.path.join(root_data_dir, 'eyetracking/roi_rect_tables')
    params['positions_dir'] = path_to_positions
    params['neural_timeline_dir'] = path_to_time_vecs
    params['pupil_size_dir'] = path_to_pupil_vecs
    params['roi_rects_dir'] = path_to_roi_rects
    logger.info("Raw data directories added to params.")
    return params


def add_paths_to_all_data_files_to_params(params):
    """
    Populates the paths to data files categorized by session, interaction type, run number, and data type.
    Parameters:
    - params (dict): Dictionary containing paths to 'positions_dir', 'neural_timeline_dir', and 'pupil_size_dir'.
    Returns:
    - params (dict): Updated dictionary with 'data_file_paths' field, which contains paths categorized by session,
      interaction type, and run number, along with a dynamic legend describing the structure.
    """
    # Define directories
    directories = {
        'positions': params['positions_dir'],
        'neural_timeline': params['neural_timeline_dir'],
        'pupil_size': params['pupil_size_dir'],
        'roi_rects': params['roi_rects_dir']
    }
    # Initialize data structure to store file paths organized by session
    paths_dict = {}
    # Define regex to extract session name (date), file type, and run number
    file_pattern = re.compile(r'(\d{8})_(position|dot)_(\d+)\.mat')
    logger.info("Populating paths to data files.")
    # Iterate over each directory
    for data_type, dir_path in directories.items():
        logger.info(f"Processing directory: {dir_path}")
        try:
            for filename in os.listdir(dir_path):
                match = file_pattern.match(filename)
                if match:
                    session_name, file_type, run_number = match.groups()
                    run_number = int(run_number)
                    # Initialize session structure if not already present
                    paths_dict.setdefault(session_name, {
                        'interactive': {},
                        'non_interactive': {}
                    })
                    # Determine interaction type based on file type
                    interaction_type = 'interactive' if file_type == 'position' else 'non_interactive'
                    # Initialize run structure if not already present
                    paths_dict[session_name][interaction_type].setdefault(run_number, {
                        'positions': None,
                        'neural_timeline': None,
                        'pupil_size': None
                    })
                    # Assign the file path to the appropriate data type
                    if data_type == 'positions':
                        paths_dict[session_name][interaction_type][run_number]['positions'] = os.path.join(dir_path, filename)
                    elif data_type == 'neural_timeline':
                        paths_dict[session_name][interaction_type][run_number]['neural_timeline'] = os.path.join(dir_path, filename)
                    elif data_type == 'pupil_size':
                        paths_dict[session_name][interaction_type][run_number]['pupil_size'] = os.path.join(dir_path, filename)
                    elif data_type == 'roi_rects':
                        paths_dict[session_name][interaction_type][run_number]['roi_rects'] = os.path.join(dir_path, filename)
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")
    # Generate a dynamic legend based on the newly structured paths dictionary
    paths_dict['legend'] = util.generate_behav_dict_legend(paths_dict)
    logger.info("Paths to all data files populated successfully.")
    # Update params with the structured paths dictionary
    params['data_file_paths'] = paths_dict
    return params


def prune_data_file_paths_with_pos_time_filename_mismatch(params):
    """
    Prunes the data file paths to ensure that the filenames of positions, neural timeline, and pupil size 
    are consistent within each run. Runs with mismatched or missing filenames are discarded and recorded.
    Parameters:
    - params (dict): Dictionary containing 'data_file_paths' with paths categorized by session, interaction type, 
      and run number.
    Returns:
    - params (dict): Updated dictionary with pruned 'data_file_paths' and a new 'discarded_paths' field 
      that records paths of discarded files.
    """
    logger.info("Pruning data file paths to ensure consistency of filenames across data types.")
    # Extract the data paths dictionary from params
    paths_dict = params.get('data_file_paths', {})
    discarded_paths = {}
    # Iterate over sessions
    for session, interaction_types in paths_dict.items():
        if session == 'legend':  # Skip the legend key
            continue
        # Initialize discarded paths for the session if not already present
        discarded_paths[session] = {'interactive': {}, 'non_interactive': {}}
        # Iterate over interaction types (interactive, non_interactive)
        for interaction_type, runs in interaction_types.items():
            for run, data_types in list(runs.items()):
                # Extract filenames for each data type
                positions_file = data_types.get('positions')
                neural_timeline_file = data_types.get('neural_timeline')
                pupil_size_file = data_types.get('pupil_size')
                # Extract just the filenames without paths
                positions_filename = positions_file.split('/')[-1] if positions_file else None
                neural_timeline_filename = neural_timeline_file.split('/')[-1] if neural_timeline_file else None
                pupil_size_filename = pupil_size_file.split('/')[-1] if pupil_size_file else None
                # Check if filenames are consistent
                filenames = [positions_filename, neural_timeline_filename, pupil_size_filename]
                if len(set(filenames)) > 1:  # Check for mismatch INCLUDING missing files
                    # Move the run to discarded paths and remove from the main paths
                    discarded_paths[session][interaction_type][run] = paths_dict[session][interaction_type].pop(run)
                    # Log which data types have inconsistent or missing filenames
                    inconsistent_types = [dtype for dtype, fname in zip(['positions', 'neural_timeline', 'pupil_size'], filenames) if fname != positions_filename]
                    logger.info(f"Discarded {interaction_type} run {run} for session {session}. Inconsistent or missing filenames in: {', '.join(inconsistent_types)}.")
    # Update params with the pruned paths and the discarded paths
    params['data_file_paths'] = paths_dict
    params['discarded_paths'] = discarded_paths
    logger.info("Data file paths pruned successfully.")
    return params


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
        for session, interaction_types in data_file_paths.items()
        if session != 'legend'
        for interaction_type, runs in interaction_types.items()
    )
    logger.info(f"Total tasks to process: {total_tasks}")
    # Choose between parallel and serial processing
    if use_parallel:
        logger.info("Using parallel processing to load data.")
        with ThreadPoolExecutor() as executor:
            futures = []
            for session, interaction_types in data_file_paths.items():
                if session == 'legend':  # Skip legend if present
                    continue
                for interaction_type, runs in interaction_types.items():
                    for run, data_types in runs.items():
                        for data_key, file_path in data_types.items():
                            if file_path:  # Ensure the file path exists
                                futures.append(executor.submit(load_run_data, data_key, session, interaction_type, run, file_path))
            # Collect the results as they complete with tqdm progress bar
            with tqdm(total=total_tasks, desc="Loading Data", unit="file", leave=False) as pbar:
                for future in as_completed(futures):
                    temp_results.append(future.result())
                    pbar.update(1)
    else:
        logger.info("Using serial processing to load data.")
        # Serial processing with a tqdm progress bar
        with tqdm(total=total_tasks, desc="Loading Data", unit="file", leave=False) as pbar:
            for session, interaction_types in data_file_paths.items():
                if session == 'legend':  # Skip legend if present
                    continue
                for interaction_type, runs in interaction_types.items():
                    for run, data_types in runs.items():
                        for data_key, file_path in data_types.items():
                            if file_path:  # Ensure the file path exists
                                result = load_run_data(data_key, session, interaction_type, run, file_path)
                                temp_results.append(result)
                            pbar.update(1)  # Manually update the progress bar
    pbar.close()  # Explicitly close the first progress bar
    # Assignment of results into the gaze_data_dict with a progress bar
    logger.info("Assigning loaded data to the gaze data dictionary.")
    with tqdm(total=len(temp_results), desc="Assigning Data", unit="assignment") as pbar:
        for session, interaction_type, run, data_key, data_value in temp_results:
            session_dict = gaze_data_dict.setdefault(session, {})
            interaction_dict = session_dict.setdefault(interaction_type, {})
            run_dict = interaction_dict.setdefault(run, {})
            if data_key in ['positions', 'pupil_size', 'roi_rects']:
                # Handle m1 and m2 under positions and pupil size
                data_subdict = run_dict.setdefault(data_key, {})
                if 'm1' in data_value:
                    data_subdict['m1'] = data_value['m1']
                if 'm2' in data_value:
                    data_subdict['m2'] = data_value['m2']
            elif data_key == 'neural_timeline':
                # Directly assign neural_timeline data
                run_dict['neural_timeline'] = data_value
            pbar.update(1)
    logger.info("Completed loading gaze data.")
    # Add a concise legend to the gaze_data_dict
    gaze_data_dict['legend'] = util.generate_behav_dict_legend(gaze_data_dict)
    return gaze_data_dict


def load_run_data(data_type, session, interaction_type, run, file_path):
    """
    Loads data based on the data type and file path provided.
    Parameters:
    - data_type (str): The type of data to load ('positions', 'neural_timeline', 'pupil_size').
    - session (str): The session identifier.
    - interaction_type (str): The type of interaction ('interactive' or 'non_interactive').
    - run (int): The run number.
    - file_path (str): The path to the data file.
    Returns:
    - tuple: A tuple containing the session, interaction type, run, data key, and the loaded data.
    """
    if data_type == 'positions':
        m1_data, m2_data = process_position_file(file_path)
        return session, interaction_type, run, 'positions', {'m1': m1_data, 'm2': m2_data}
    elif data_type == 'neural_timeline':
        t_data = process_time_file(file_path)
        return session, interaction_type, run, 'neural_timeline', t_data
    elif data_type == 'pupil_size':
        m1_data, m2_data = process_pupil_file(file_path)
        return session, interaction_type, run, 'pupil_size', {'m1': m1_data, 'm2': m2_data}
    elif data_type == 'roi_rects':
        m1_data, m2_data = process_roi_rects_file(file_path)
        return session, interaction_type, run, 'roi_rects', {'m1': m1_data, 'm2': m2_data}
    # Fallback if the data type does not match expected values
    logger.warning(f"Unexpected data type '{data_type}' for session {session}, interaction {interaction_type}, run {run}.")
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
        return m1_data, m2_data
    # Log and set a breakpoint if data is missing or improperly formatted
    logger.warning(f"Position data is missing or improperly formatted in file: {mat_file}")
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
        return m1_data, m2_data
    # Log and set a breakpoint if data is missing or improperly formatted
    logger.warning(f"Pupil size data is missing or improperly formatted in file: {mat_file}")
    return None, None


def process_roi_rects_file(mat_file):
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

        # Helper function to extract ROI data
        def extract_roi_dict(agent_roi_data):
            if agent_roi_data is None:
                return None
            roi_dict = {}
            for roi_name in agent_roi_data.dtype.names:
                roi_dict[roi_name] = agent_roi_data[roi_name][0][0][0]  # Adjust based on the data structure
            return roi_dict

        # Extract m1 and m2 data
        m1_data = extract_roi_dict(roi_data['m1']) if 'm1' in roi_data.dtype.names else None
        m2_data = extract_roi_dict(roi_data['m2']) if 'm2' in roi_data.dtype.names else None
        return m1_data, m2_data
    # Log if roi_data is missing or improperly formatted
    logger.warning(f"ROI rects data is missing or improperly formatted in file: {mat_file}")
    return None, None



def clean_and_log_missing_dict_leaves(data_dict):
    """
    Recursively checks and removes keys with missing or empty data in a nested dictionary, logs missing paths.
    Parameters:
    - data_dict (dict): The dictionary to clean.
    Returns:
    - pruned_dict (dict): The cleaned dictionary with missing or empty values removed.
    - missing_paths (list): List of paths where data was missing or empty.
    """
    logger.info(f"Removing dict keys with empty values (m2 data in some sessions/measurement)")
    missing_paths = []
    total_paths_checked = 0
    pruned_dict, total_paths_checked, missing_paths = _remove_and_log_missing_leaves(
        data_dict, missing_paths, total_paths_checked
    )
    # Log the summary of the operation
    logger.info(f"Total paths checked: {total_paths_checked}")
    logger.info(f"Total paths removed: {len(missing_paths)}")
    return pruned_dict, missing_paths


def _remove_and_log_missing_leaves(d, missing_paths, total_paths_checked, path=""):
    """
    Helper function that recursively removes keys with missing or empty data, logs paths of removed items.
    Parameters:
    - d (dict): The dictionary to clean.
    - missing_paths (list): List to accumulate paths of missing or empty data.
    - total_paths_checked (int): Counter to keep track of the total paths checked.
    - path (str): Current path of the dictionary during recursion.
    Returns:
    - d (dict): The cleaned dictionary with missing or empty values removed.
    - total_paths_checked (int): Updated count of paths checked.
    - missing_paths (list): Updated list of paths where data was missing or empty.
    """
    for key in list(d.keys()):  # Use list(d.keys()) to avoid runtime error during iteration if keys are removed
        current_path = f"{path}/{key}" if path else str(key)
        value = d[key]
        total_paths_checked += 1  # Increment the counter for each path checked
        # Check for missing or empty values before recursion
        if value is None or (hasattr(value, "size") and value.size == 0):
            missing_paths.append(current_path)
            d.pop(key)  # Directly remove the key
        elif isinstance(value, dict):
            # Recursively clean sub-dictionaries
            d[key], total_paths_checked, missing_paths = _remove_and_log_missing_leaves(
                value, missing_paths, total_paths_checked, current_path
            )
            # If the sub-dictionary is empty after recursion, remove it
            if not d[key]:  # This catches sub-dicts that became empty after cleaning
                logger.warning(f"Empty sub-dictionary at path: {current_path}")
                missing_paths.append(current_path)
                d.pop(key)
    return d, total_paths_checked, missing_paths


def prune_nan_values_in_timeseries(params, gaze_data_dict):
    """
    Prunes NaN values from the time series in the gaze data dictionary and 
    adjusts positions and pupil_size for m1 and m2 (if present) accordingly.
    The pruned dictionary is saved to the processed data directory specified in params.
    Parameters:
    - gaze_data_dict (dict): The gaze data dictionary containing session, interaction type, and run data.
    - params (dict): A dictionary containing configuration parameters, including the processed data directory path.
    Returns:
    - nan_removed_gaze_data_dict (dict): The pruned gaze data dictionary with NaN values removed.
    """
    # Create a copy of the gaze data dictionary to store the pruned version
    nan_removed_gaze_data_dict = {}
    # Iterate over the original gaze data dictionary
    for session, session_dict in gaze_data_dict.items():
        if session == 'legend':  # Skip the legend entry
            continue
        pruned_session_dict = {}
        for interaction_type, interaction_dict in session_dict.items():
            pruned_interaction_dict = {}
            for run, run_dict in interaction_dict.items():
                # Extract the neural timeline (time series)
                time_series = run_dict.get('neural_timeline')
                if time_series is not None:
                    # Prune NaN values and adjust corresponding time series using the helper function
                    pruned_positions, pruned_pupil_size, pruned_time_series = prune_nans_in_specific_timeseries(
                        time_series,
                        run_dict.get('positions', {}),
                        run_dict.get('pupil_size', {})
                    )
                    # Create a new run dictionary with pruned data
                    pruned_run_dict = {
                        'positions': pruned_positions,
                        'pupil_size': pruned_pupil_size,
                        'neural_timeline': pruned_time_series,
                        'roi_rects': run_dict.get('roi_rects')
                    }
                    pruned_interaction_dict[run] = pruned_run_dict
            pruned_session_dict[interaction_type] = pruned_interaction_dict
        nan_removed_gaze_data_dict[session] = pruned_session_dict
    return nan_removed_gaze_data_dict


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
    valid_indices = valid_time_indices.copy()  # Initialize valid indices with time series indices
    # Adjust valid indices based on positions and pupil sizes for m1 and m2
    for key in ['m1', 'm2']:
        # Adjust for positions[key] if present, not None, and non-empty
        if key in positions and positions[key] is not None and positions[key].size > 0:
            # Ensure positions[key] is 2D; adjust valid_indices accordingly
            position_valid_indices = ~np.isnan(positions[key]).any(axis=0)
            valid_indices = valid_indices[:len(position_valid_indices)] & position_valid_indices
        # Adjust for pupil_size[key] if present, not None, and non-empty
        if key in pupil_size and pupil_size[key] is not None and pupil_size[key].size > 0:
            # Ensure pupil_size[key] is correctly indexed along its second axis
            pupil_valid_indices = ~np.isnan(pupil_size[key][0, :])  # Index columns correctly
            valid_indices = valid_indices[:len(pupil_valid_indices)] & pupil_valid_indices
    # Prune the time series based on combined valid indices
    pruned_time_series = time_series[valid_indices]
    # Prune positions for m1 and m2 based on the combined valid indices
    pruned_positions = {
        key: positions[key][:, valid_indices] 
        for key in ['m1', 'm2'] 
        if key in positions and positions[key] is not None and positions[key].size > 0 and valid_indices.size > 0
    }
    # Prune pupil size for m1 and m2 based on the combined valid indices
    pruned_pupil_size = {
        key: pupil_size[key][:, valid_indices] 
        for key in ['m1', 'm2'] 
        if key in pupil_size and pupil_size[key] is not None and pupil_size[key].size > 0 and valid_indices.size > 0
    }
    return pruned_positions, pruned_pupil_size, pruned_time_series



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
    # Merge the results from fixations and saccades
    for session in fixation_vectors:
        behavioral_vectors_dict.setdefault(session, {}).update(fixation_vectors[session])
    for session in saccade_vectors:
        behavioral_vectors_dict.setdefault(session, {}).update(saccade_vectors[session])
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
                    behavior_indices = agent_data[f'{behavior_type}index']
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




