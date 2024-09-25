#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:04:31 2024

@author: pg496
"""

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

