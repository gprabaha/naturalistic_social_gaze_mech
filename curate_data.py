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
    params['positions_dir'] = path_to_positions
    params['neural_timeline_dir'] = path_to_time_vecs
    params['pupil_size_dir'] = path_to_pupil_vecs
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
        'pupil_size': params['pupil_size_dir']
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
                    if session_name not in paths_dict:
                        paths_dict[session_name] = {
                            'interactive': {},
                            'non_interactive': {}
                        }
                    # Determine interaction type based on file type
                    interaction_type = 'interactive' if file_type == 'position' else 'non_interactive'
                    # Initialize run structure if not already present
                    if run_number not in paths_dict[session_name][interaction_type]:
                        paths_dict[session_name][interaction_type][run_number] = {
                            'positions': None,
                            'neural_timeline': None,
                            'pupil_size': None
                        }
                    # Assign the file path to the appropriate data type
                    if data_type == 'positions' and file_type == 'position':
                        paths_dict[session_name][interaction_type][run_number]['positions'] = os.path.join(dir_path, filename)
                    elif data_type == 'neural_timeline':
                        paths_dict[session_name][interaction_type][run_number]['neural_timeline'] = os.path.join(dir_path, filename)
                    elif data_type == 'pupil_size':
                        paths_dict[session_name][interaction_type][run_number]['pupil_size'] = os.path.join(dir_path, filename)
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")
    # Generate a dynamic legend based on the newly structured paths dictionary
    paths_dict['legend'] = util.generate_behav_dict_legend(paths_dict)
    logger.info("Paths to all data files populated successfully.")
    # Update params with the structured paths dictionary
    params['data_file_paths'] = paths_dict
    return params


def prune_data_file_paths(params):
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
                if len(set(filenames) - {None}) > 1:  # Check for mismatch ignoring missing files
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
            for future in tqdm(as_completed(futures), total=total_tasks, desc="Loading Data", unit="file"):
                temp_results.append(future.result())
    else:
        logger.info("Using serial processing to load data.")
        # Serial processing with a tqdm progress bar
        with tqdm(total=total_tasks, desc="Loading Data", unit="file") as pbar:
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
    # Assignment of results into the gaze_data_dict with a progress bar
    logger.info("Assigning loaded data to the gaze data dictionary.")
    with tqdm(total=len(temp_results), desc="Assigning Data", unit="assignment") as pbar:
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
            pbar.update(1)
    logger.info("Completed loading gaze data.")
    # Add a concise legend to the gaze_data_dict
    gaze_data_dict['legend'] = util.generate_behav_dict_legend(gaze_data_dict)
    # Check the final structure and report any missing data
    missing_data_dict_paths, total_data_paths = util.check_dict_leaves(gaze_data_dict)
    if missing_data_dict_paths:
        logger.warning(f"Missing or empty data found at {len(missing_data_dict_paths)} data dict paths out of {total_data_paths} total paths.")
    else:
        logger.info(f"All data leaves are correctly populated. Total paths checked: {total_data_paths}.")
    return gaze_data_dict, missing_data_dict_paths


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
        return session, interaction_type, run, 'neural_timeline', t_data  # Updated key to match the correct naming
    elif data_type == 'pupil_size':
        m1_data, m2_data = process_pupil_file(file_path)
        return session, interaction_type, run, 'pupil_size', {'m1': m1_data, 'm2': m2_data}
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


def prune_nan_values_in_timeseries(gaze_data_dict):
    """
    Prunes NaN values from the time series in the gaze data dictionary and 
    adjusts positions and pupil_size for m1 and m2 (if present) accordingly.
    The pruned dictionary is returned as `nan_removed_gaze_data_dict`.
    Parameters:
    - gaze_data_dict (dict): The gaze data dictionary containing session, interaction type, and run data.
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
                        'neural_timeline': pruned_time_series
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
    pruned_positions = {key: positions[key][:, valid_indices[:positions[key].shape[1]]] if key in positions and positions[key] is not None and positions[key].size > 0 else np.array([])
                        for key in ['m1', 'm2']}
    # Prune pupil size for m1 and m2 based on the combined valid indices
    pruned_pupil_size = {key: pupil_size[key][:, valid_indices] if key in pupil_size and pupil_size[key] is not None and pupil_size[key].size > 0 else np.array([])
                         for key in ['m1', 'm2']}
    return pruned_positions, pruned_pupil_size, pruned_time_series