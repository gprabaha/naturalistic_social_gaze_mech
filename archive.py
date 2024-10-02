#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:04:31 2024

@author: pg496
"""




## from curate data: old dictionary structure, now using dataframe

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


## From filter_behav.py

import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import re
import load_data
import util

def get_gaze_timepos_across_sessions(params):
    root_data_dir = params.get('root_data_dir')
    use_parallel = params.get('use_parallel', False)
    intermediates_dir = os.path.join(root_data_dir, 'intermediates')
    os.makedirs(intermediates_dir, exist_ok=True)
    pos_pattern = r"(\d{8})_position_(\d+).mat"
    time_pattern = r"(\d{8})_position_(\d+).mat"  # Assuming the pattern is the same; adjust if different
    path_to_positions = os.path.join(root_data_dir, 'eyetracking/aligned_raw_samples/position')
    path_to_time_vecs = os.path.join(root_data_dir, 'eyetracking/aligned_raw_samples/time')
    pos_mat_files_sorted = util.get_sorted_files(path_to_positions, pos_pattern)
    time_mat_files_sorted = util.get_sorted_files(path_to_time_vecs, time_pattern)
    sorted_position_path_list = [os.path.join(path_to_positions, f) for f in pos_mat_files_sorted]
    sorted_time_path_list = [os.path.join(path_to_time_vecs, f) for f in time_mat_files_sorted]
    m1_positions = [None] * len(sorted_position_path_list)
    m2_positions = [None] * len(sorted_position_path_list)
    time_vectors = [None] * len(sorted_time_path_list)
    valid_pos_files = [None] * len(sorted_position_path_list)
    valid_time_files = [None] * len(sorted_time_path_list)

    session_infos = [{'session_name': re.match(pos_pattern, os.path.basename(f)).group(1), 'run_number': int(re.match(pos_pattern, os.path.basename(f)).group(2))} for f in sorted_position_path_list]

    if use_parallel:
        print('Loading pos-time files in parallel')
        m1_positions, m2_positions, valid_pos_files, \
            time_vectors, valid_time_files = process_postime_files_concurrently(
                sorted_position_path_list, sorted_time_path_list,
                process_pos_file, process_time_file, session_infos)
    else:
        print('Loading pos-time files in serial')
        m1_positions, m2_positions, valid_pos_files, \
            time_vectors, valid_time_files = process_postime_files_sequentially(
                sorted_position_path_list, sorted_time_path_list,
                process_pos_file, process_time_file, session_infos)

    # Convert dictionaries to lists based on sorted indices
    m1_positions_list = [m1_positions[i] for i in sorted(m1_positions.keys()) if m1_positions[i] is not None]
    m2_positions_list = [m2_positions[i] for i in sorted(m2_positions.keys()) if m2_positions[i] is not None]
    time_vectors_list = [time_vectors[i] for i in sorted(time_vectors.keys()) if time_vectors[i] is not None]
    sorted_position_path_list = [sorted_position_path_list[i] for i in sorted(valid_pos_files.keys()) if valid_pos_files[i] is not None]
    sorted_time_path_list = [sorted_time_path_list[i] for i in sorted(valid_time_files.keys()) if valid_time_files[i] is not None]

    m1_positions_list, m2_positions_list, time_vectors_list, \
        sorted_position_path_list, sorted_time_path_list = \
            util.filter_none_entries(
                m1_positions_list, m2_positions_list, time_vectors_list,
                sorted_position_path_list, sorted_time_path_list)
    
    # Synchronize the file lists
    sorted_time_path_list, sorted_position_path_list, \
        m1_positions_list, m2_positions_list, time_vectors_list = util.synchronize_file_lists(
            sorted_time_path_list, sorted_position_path_list,
            m1_positions_list, m2_positions_list, time_vectors_list)
    
    # Check if filenames match
    pos_filenames = [os.path.basename(path) for path in sorted_position_path_list]
    time_filenames = [os.path.basename(path) for path in sorted_time_path_list]
    mismatched_pos_files = [file for file in pos_filenames if file not in time_filenames]
    mismatched_time_files = [file for file in time_filenames if file not in pos_filenames]
    if mismatched_pos_files or mismatched_time_files:
        print(f"Mismatched position files: {mismatched_pos_files}")
        print(f"Mismatched time files: {mismatched_time_files}")
        raise ValueError("Mismatch between valid position files and valid time files")
    util.save_arrays_as_npy(intermediates_dir, m1_positions_list, 'm1_positions')
    util.save_arrays_as_npy(intermediates_dir, m2_positions_list, 'm2_positions')
    util.save_arrays_as_npy(intermediates_dir, time_vectors_list, 'time_vectors')
    path_list_file = os.path.join(intermediates_dir, 'sorted_position_path_list.txt')
    time_list_file = os.path.join(intermediates_dir, 'sorted_time_path_list.txt')
    with open(path_list_file, 'w') as f:
        f.write('\n'.join(sorted_position_path_list))
    with open(time_list_file, 'w') as f:
        f.write('\n'.join(sorted_time_path_list))
    return sorted_position_path_list, m1_positions_list, m2_positions_list, sorted_time_path_list, time_vectors_list

def process_postime_files_concurrently(sorted_position_path_list,
                                       sorted_time_path_list,
                                       process_pos_file, process_time_file,
                                       session_infos):
    m1_positions = {}
    m2_positions = {}
    valid_pos_files = {}
    time_vectors = {}
    valid_time_files = {}
    with ThreadPoolExecutor() as executor:
        pos_futures = [executor.submit(process_pos_file, i, mat_file, session_infos[i])
                       for i, mat_file in enumerate(sorted_position_path_list)]
        time_futures = [executor.submit(process_time_file, i, mat_file, session_infos[i])
                        for i, mat_file in enumerate(sorted_time_path_list)]
        for future in tqdm(concurrent.futures.as_completed(pos_futures),
                           total=len(pos_futures), desc='n files loaded'):
            index, m1_pos, m2_pos, valid_file, session_info = future.result()
            if m1_pos is not None and m2_pos is not None:
                m1_positions[index] = m1_pos
                m2_positions[index] = m2_pos
                valid_pos_files[index] = valid_file
        for future in tqdm(concurrent.futures.as_completed(time_futures),
                           total=len(time_futures), desc='n time files loaded'):
            index, t, valid_file, session_info = future.result()
            if t is not None:
                time_vectors[index] = t
                valid_time_files[index] = valid_file
    return m1_positions, m2_positions, valid_pos_files, time_vectors, valid_time_files

def process_postime_files_sequentially(sorted_position_path_list,
                                       sorted_time_path_list,
                                       process_pos_file, process_time_file,
                                       session_infos):
    m1_positions = {}
    m2_positions = {}
    valid_pos_files = {}
    time_vectors = {}
    valid_time_files = {}
    for i, mat_file in tqdm(enumerate(sorted_position_path_list),
                            total=len(sorted_position_path_list), desc='n files loaded'):
        index, m1_pos, m2_pos, valid_file, session_info = process_pos_file(i, mat_file, session_infos[i])
        if m1_pos is not None and m2_pos is not None:
            m1_positions[index] = m1_pos
            m2_positions[index] = m2_pos
            valid_pos_files[index] = valid_file
    for i, mat_file in tqdm(enumerate(sorted_time_path_list),
                            total=len(sorted_time_path_list), desc='n time files loaded'):
        index, t, valid_file, session_info = process_time_file(i, mat_file, session_infos[i])
        if t is not None:
            time_vectors[index] = t
            valid_time_files[index] = valid_file
    return m1_positions, m2_positions, valid_pos_files, time_vectors, valid_time_files

def process_pos_file(index, mat_file, session_info):
    mat_data = load_data.load_mat_from_path(mat_file)
    if 'var' in mat_data:
        aligned_positions = mat_data['var'][0][0]
        if 'm1' in aligned_positions.dtype.names and 'm2' in aligned_positions.dtype.names:
            return (index, tuple(aligned_positions['m1'].T), tuple(aligned_positions['m2'].T), mat_file, session_info)
    return (index, None, None, None, session_info)

def process_time_file(index, mat_file, session_info):
    mat_data = load_data.load_mat_from_path(mat_file)
    if 'var' in mat_data:
        t = mat_data['var'][0][0]['t']
        return (index, t, mat_file, session_info)
    return (index, None, None, session_info)



## From old load_data

def get_combined_gaze_pos_and_time_lists(params):
    """
    Load combined gaze position and time lists.
    Args:
    - params (dict): Dictionary containing parameters.
    Returns:
    - sorted_position_path_list (list): List of sorted file paths.
    - m1_positions (list): List of m1 gaze positions.
    - m2_positions (list): List of m2 gaze positions.
    - time_vectors (list): List of time vectors.
    """
    root_data_dir = params.get('root_data_dir')
    intermediates_dir = os.path.join(root_data_dir, 'intermediates')
    # Load sorted position and time file lists
    sorted_position_path_list_file = os.path.join(intermediates_dir, 'sorted_position_path_list.txt')
    sorted_time_path_list_file = os.path.join(intermediates_dir, 'sorted_time_path_list.txt')
    with open(sorted_position_path_list_file, 'r') as f:
        sorted_position_path_list = f.read().splitlines()
    with open(sorted_time_path_list_file, 'r') as f:
        sorted_time_path_list = f.read().splitlines()
    # Load m1_positions, m2_positions, and time_vectors as lists of arrays
    m1_positions = load_arrays_from_multiple_npy(intermediates_dir, 'm1_positions')
    m2_positions = load_arrays_from_multiple_npy(intermediates_dir, 'm2_positions')
    time_vectors = load_arrays_from_multiple_npy(intermediates_dir, 'time_vectors')
    return sorted_position_path_list, m1_positions, m2_positions, sorted_time_path_list, time_vectors


def load_arrays_from_multiple_npy(directory, filename_prefix):
    """
    Load arrays from .npy files with preserved list structure.
    Args:
    - directory (str): Directory containing the files.
    - filename_prefix (str): Prefix for the input filenames.
    Returns:
    - arrays (list): List of loaded arrays.
    """
    arrays = []
    i = 0
    while True:
        input_file = os.path.join(directory, f"{filename_prefix}_{i}.npy")
        if not os.path.exists(input_file):
            break
        arrays.append(np.load(input_file))
        i += 1
    return arrays





## From data_manager.py

def print_dict_structure(d, indent=0):
    """
    Recursively prints the structure of a nested dictionary without printing the values.

    :param d: The dictionary to explore.
    :param indent: The current level of indentation (used for nested dictionaries).
    """
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + str(key) + ':')
            print_dict_structure(value, indent + 1)
    elif isinstance(d, list):
        print('  ' * indent + '[List]')
    else:
        print('  ' * indent + '[Value]')


## From old data manager/ analytze gaze signals

if params.get('extract_postime_from_mat_files', False):
    sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
        filter_behav.get_gaze_timepos_across_sessions(params)
else:
    sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
        load_data.get_combined_gaze_pos_and_time_lists(params)

print(sorted_position_path_list)


pos_pattern = r"(\d{8})_position_(\d+).mat"
session_infos = [{'session_name': re.match(pos_pattern, os.path.basename(f)).group(1), 'run_number': int(re.match(pos_pattern, os.path.basename(f)).group(2))} for f in sorted_position_path_list]

# Extract fixations and saccades for both monkeys
fixations_m1, saccades_m1, fixations_m2, saccades_m2 = fix_and_saccades.extract_fixations_for_both_monkeys(params, m1_gaze_positions, m2_gaze_positions, time_vecs, session_infos)



# old fix_and_saccades.py

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging

import pdb

import util
from fix_saccade_detectors import ClusterFixationDetector, EyeMVMFixationDetector, EyeMVMSaccadeDetector

def extract_fixations_for_both_monkeys(params, m1_gaze_positions, m2_gaze_positions, time_vecs, session_infos):
    """
    Extracts fixations and saccades for both monkeys.
    Parameters:
    - params (dict): Dictionary containing parameters.
    - m1_gaze_positions (list): List of m1 gaze positions arrays.
    - m2_gaze_positions (list): List of m2 gaze positions arrays.
    - session_infos (list): List of session info dictionaries.
    Returns:
    - fixations_m1 (list): List of m1 fixations.
    - saccades_m1 (list): List of m1 saccades.
    - fixations_m2 (list): List of m2 fixations.
    - saccades_m2 (list): List of m2 saccades.
    """
    sessions_data_m1 = [(positions, info, time_vec, params) for positions, time_vec, info in zip(m1_gaze_positions, time_vecs, session_infos)]
    sessions_data_m2 = [(positions, info, time_vec, params) for positions, time_vec, info in zip(m2_gaze_positions, time_vecs, session_infos)]

    fixations_m1, saccades_m1 = extract_fixations_and_saccades(sessions_data_m1, params['use_parallel'])
    fixations_m2, saccades_m2 = extract_fixations_and_saccades(sessions_data_m2, params['use_parallel'])

    all_fix_timepos_m1 = process_fixation_results(fixations_m1)
    all_fix_timepos_m2 = process_fixation_results(fixations_m2)

    save_fixation_and_saccade_results(params['intermediates'], all_fix_timepos_m1, fixations_m1, saccades_m1, params, 'm1')
    save_fixation_and_saccade_results(params['intermediates'], all_fix_timepos_m2, fixations_m2, saccades_m2, params, 'm2')

    return fixations_m1, saccades_m1, fixations_m2, saccades_m2


'''
Edit here. We are trying to create a timevec like we did for OTNAL, but for this 
dataset a corresponding timevec already exists. we might have to do some NaN
removal from both position and time before fixations or saccades can be detected
'''

def extract_fixations_and_saccades(sessions_data, use_parallel):
    """
    Extracts fixations and saccades from session data.
    Parameters:
    - sessions_data (list): List of session data tuples.
    - use_parallel (bool): Flag to determine if parallel processing should be used.
    Returns:
    - fix_detection_results (list): List of fixation detection results.
    - saccade_detection_results (list): List of saccade detection results.
    """
    if use_parallel:
        print("\nExtracting fixations and saccades in parallel")
        num_cores = multiprocessing.cpu_count()
        num_processes = min(num_cores, len(sessions_data))
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(get_session_fixations_and_saccades, session_data): session_data for session_data in sessions_data}
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error processing session data: {e}")
                    continue
    else:
        print("\nExtracting fixations and saccades serially")
        results = [get_session_fixations_and_saccades(session_data) for session_data in sessions_data]
    fix_detection_results, saccade_detection_results = zip(*results)
    return fix_detection_results, saccade_detection_results

def get_session_fixations_and_saccades(session_data):
    """
    Extracts fixations and saccades for a session.
    Parameters:
    - session_data (tuple): Tuple containing session identifier, positions, and metadata.
    Returns:
    - fix_timepos_df (pd.DataFrame): DataFrame of fixation time positions.
    - info (dict): Metadata information for the session.
    - session_saccades (list): List of saccades for the session.
    """
    positions, info, time_vec, params = session_data
    session_name = info['session_name']
    sampling_rate = params.get('sampling_rate', 0.001)
    n_samples = positions.shape[0]
    nan_removed_positions, nan_removed_time_vec = util.remove_nans(positions, time_vec)
    if params.get('fixation_detection_method', 'default') == 'cluster_fix':
        detector = ClusterFixationDetector(samprate=sampling_rate)
        x_coords = nan_removed_positions[:, 0]
        y_coords = nan_removed_positions[:, 1]
        # Transform into the expected format
        eyedat = [(x_coords, y_coords)]
        fix_stats = detector.detect_fixations(eyedat)
        fixationtimes = fix_stats[0]['fixationtimes']
        fixations = fix_stats[0]['fixations']
        saccadetimes = fix_stats[0]['saccadetimes']
        saccades = format_saccades(saccadetimes, nan_removed_positions, info)
    else:
        fix_detector = EyeMVMFixationDetector(sampling_rate=sampling_rate)
        fixationtimes, fixations = fix_detector.detect_fixations(nan_removed_positions, nan_removed_time_vec, session_name)
        saccade_detector = EyeMVMSaccadeDetector(params['vel_thresh'], params['min_samples'], params['smooth_func'])
        saccades = saccade_detector.extract_saccades_for_session((nan_removed_positions, nan_removed_time_vec, info))

    fix_timepos_df = pd.DataFrame({
        'start_time': fixationtimes[0],
        'end_time': fixationtimes[1],
        'fix_x': fixations[0],
        'fix_y': fixations[1]
    })
    return fix_timepos_df, info, saccades

def process_fixation_results(fix_detection_results):
    """
    Processes the results from fixation detection.
    Parameters:
    - fix_detection_results (list): List of fixation detection results.
    Returns:
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    """
    all_fix_timepos = pd.DataFrame()
    for session_timepos_df, _ in fix_detection_results:
        all_fix_timepos = pd.concat([all_fix_timepos, session_timepos_df], ignore_index=True)
    return all_fix_timepos

def save_fixation_and_saccade_results(processed_data_dir, all_fix_timepos, fix_detection_results, saccade_detection_results, params, monkey):
    """
    Saves fixation and saccade results to files.
    Parameters:
    - processed_data_dir (str): Directory to save processed data.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    - saccade_detection_results (list): List of saccade detection results.
    - params (dict): Dictionary of parameters.
    - monkey (str): 'm1' or 'm2' to specify monkey data.
    """
    flag_info = util.get_filename_flag_info(params)
    timepos_file_name = f'fix_timepos_{monkey}{flag_info}.csv'
    all_fix_timepos.to_csv(os.path.join(processed_data_dir, timepos_file_name), index=False)
    # Save the fixation and saccade detection results using pickle or similar method
    fix_detection_file_name = f'fix_detection_{monkey}{flag_info}.pkl'
    saccade_detection_file_name = f'saccade_detection_{monkey}{flag_info}.pkl'
    util.save_to_pickle(os.path.join(processed_data_dir, fix_detection_file_name), fix_detection_results)
    util.save_to_pickle(os.path.join(processed_data_dir, saccade_detection_file_name), saccade_detection_results)

def format_saccades(saccadetimes, positions, info):
    """
    Formats the saccade times into a list of saccade details.
    Parameters:
    - saccadetimes (array): Array of saccade times.
    - positions (array): Array of gaze positions.
    - info (dict): Dictionary of session information.
    Returns:
    - saccades (list): List of saccade details.
    """
    # Placeholder function
    return []

def determine_roi_of_coord(position, bbox_corners):
    # Placeholder for ROI determination logic
    return 'roi_placeholder'

def determine_block(start_time, end_time, startS, stopS):
    # Placeholder for run and block determining logic based on the new dataset structure
    return 'block_placeholder'


## Old util.py


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

