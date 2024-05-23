#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:04:31 2024

@author: pg496
"""

import re
import os
import scipy
import numpy as np
import concurrent.futures
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import load_data
import util

import pdb


def get_gaze_timepos_across_sessions(params):
    """
    Get gaze positions across sessions.
    Args:
    - params (dict): Dictionary containing parameters.
    Returns:
    - sorted_position_path_list (list): List of sorted file paths.
    - m1_positions (list): List of m1 gaze positions.
    - m2_positions (list): List of m2 gaze positions.
    - sorted_time_path_list (list): List of sorted time file paths.
    - time_vectors (list): List of time vectors.
    """
    root_data_dir = params.get('root_data_dir')
    use_parallel = params.get('use_parallel', False)
    intermediates_dir = os.path.join(root_data_dir, 'intermediates')
    os.makedirs(intermediates_dir, exist_ok=True)
    pos_pattern = r"(\d{8})_positions_(\d+).mat"
    time_pattern = r"(\d{8})_positions_(\d+).mat"  # Assuming the pattern is the same; adjust if different
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
    if use_parallel:
        print('Loading pos-time files in parallel')
        m1_positions, m2_positions, valid_pos_files, \
            time_vectors, valid_time_files = process_postime_files_concurrently(
                sorted_position_path_list, sorted_time_path_list,
                process_pos_file, process_time_file)
    else:
        print('Loading pos-time files in serial')
        m1_positions, m2_positions, valid_pos_files, \
            time_vectors, valid_time_files = process_postime_files_sequentially(
                sorted_position_path_list, sorted_time_path_list,
                process_pos_file, process_time_file)
    m1_positions, m2_positions, time_vectors, \
        sorted_position_path_list, sorted_time_path_list = \
            util.filter_none_entries(
                m1_positions, m2_positions, time_vectors,
                valid_pos_files, valid_time_files)
    m1_positions, m2_positions, time_vectors, \
        sorted_position_path_list, sorted_time_path_list = \
            util.filter_none_entries(m1_positions, m2_positions, time_vectors,
                                     valid_pos_files, valid_time_files)
    # Synchronize the file lists
    sorted_time_path_list, sorted_position_path_list, \
        m1_positions, m2_positions, time_vectors = util.synchronize_file_lists(
            sorted_time_path_list, sorted_position_path_list,
            m1_positions, m2_positions, time_vectors)
    # Check if filenames match
    pos_filenames = [os.path.basename(path) for path in sorted_position_path_list]
    time_filenames = [os.path.basename(path) for path in sorted_time_path_list]
    mismatched_pos_files = [file for file in pos_filenames if file not in time_filenames]
    mismatched_time_files = [file for file in time_filenames if file not in pos_filenames]
    if mismatched_pos_files or mismatched_time_files:
        print(f"Mismatched position files: {mismatched_pos_files}")
        print(f"Mismatched time files: {mismatched_time_files}")
        raise ValueError("Mismatch between valid position files and valid time files")
    util.save_arrays_as_npy(intermediates_dir, m1_positions, 'm1_positions')
    util.save_arrays_as_npy(intermediates_dir, m2_positions, 'm2_positions')
    util.save_arrays_as_npy(intermediates_dir, time_vectors, 'time_vectors')
    path_list_file = os.path.join(intermediates_dir, 'sorted_position_path_list.txt')
    time_list_file = os.path.join(intermediates_dir, 'sorted_time_path_list.txt')
    with open(path_list_file, 'w') as f:
        f.write('\n'.join(sorted_position_path_list))
    with open(time_list_file, 'w') as f:
        f.write('\n'.join(sorted_time_path_list))
    return sorted_position_path_list, m1_positions, m2_positions, sorted_time_path_list, time_vectors


def process_postime_files_concurrently(sorted_position_path_list,
                                       sorted_time_path_list,
                                       process_pos_file, process_time_file):
    m1_positions = {}
    m2_positions = {}
    valid_pos_files = {}
    time_vectors = {}
    valid_time_files = {}
    with ThreadPoolExecutor() as executor:
        pos_futures = [executor.submit(process_pos_file, i, mat_file)
                       for i, mat_file in enumerate(sorted_position_path_list)]
        time_futures = [executor.submit(process_time_file, i, mat_file)
                        for i, mat_file in enumerate(sorted_time_path_list)]
        for future in tqdm(concurrent.futures.as_completed(pos_futures),
                           total=len(pos_futures), desc='n files loaded'):
            index, m1_pos, m2_pos, valid_file = future.result()
            if m1_pos is not None and m2_pos is not None:
                m1_positions[index] = m1_pos
                m2_positions[index] = m2_pos
                valid_pos_files[index] = valid_file
        for future in tqdm(concurrent.futures.as_completed(time_futures),
                           total=len(time_futures), desc='n time files loaded'):
            index, t, valid_file = future.result()
            if t is not None:
                time_vectors[index] = t
                valid_time_files[index] = valid_file
    return m1_positions, m2_positions, valid_pos_files, time_vectors, valid_time_files


def process_postime_files_sequentially(sorted_position_path_list,
                               sorted_time_path_list,
                               process_pos_file, process_time_file):
    m1_positions = {}
    m2_positions = {}
    valid_pos_files = {}
    time_vectors = {}
    valid_time_files = {}
    for i, mat_file in tqdm(enumerate(sorted_position_path_list),
                            total=len(sorted_position_path_list), desc='n files loaded'):
        index, m1_pos, m2_pos, valid_file = process_pos_file(i, mat_file)
        if m1_pos is not None and m2_pos is not None:
            m1_positions[index] = m1_pos
            m2_positions[index] = m2_pos
            valid_pos_files[index] = valid_file

    for i, mat_file in tqdm(enumerate(sorted_time_path_list),
                            total=len(sorted_time_path_list), desc='n time files loaded'):
        index, t, valid_file = process_time_file(i, mat_file)
        if t is not None:
            time_vectors[index] = t
            valid_time_files[index] = valid_file
    return m1_positions, m2_positions, valid_pos_files, time_vectors, valid_time_files


def process_pos_file(index, mat_file):
    mat_data = load_data.load_mat_from_path(mat_file)
    if 'var' in mat_data:
        aligned_positions = mat_data['var'][0][0]
        if 'm1' in aligned_positions.dtype.names and 'm2' in aligned_positions.dtype.names:
            return (index, tuple(aligned_positions['m1'].T), tuple(aligned_positions['m2'].T), mat_file)
    return (index, None, None, None)


def process_time_file(index, mat_file):
    mat_data = load_data.load_mat_from_path(mat_file)
    if 'var' in mat_data:
        t = mat_data['var']
        return (index, t, mat_file)
    return (index, None, None)


def extract_fixations_for_both_monkeys(params):
    sorted_position_path_list = params.get('sorted_position_path_list', [])
    m1_gaze_positions = params.get('m1_gaze_positions', [])
    m2_gaze_positions = params.get('m2_gaze_positions', [])
    
    for gaze_pos_within_run_m1, gaze_pos_within_run_m2 in zip(m1_gaze_positions, m2_gaze_positions):
        x=1
    
    return 0




