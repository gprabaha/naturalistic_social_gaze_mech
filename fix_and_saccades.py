#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:34:44 2024

@author: pg496
"""

import logging
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import interp1d

import hpc_fix_and_saccade_detector
import fixation_detector_class
import saccade_detector_class

import pdb

logger = logging.getLogger(__name__)


def detect_fixations_and_saccades(nan_removed_gaze_data_df, params):
    """
    Detects fixations and saccades in the gaze data, returning dataframes for fixations and saccades.
    Parameters:
    - nan_removed_gaze_data_df (pd.DataFrame): The DataFrame containing pruned gaze data.
    - params (dict): Configuration parameters, including paths and processing options.
    Returns:
    - fixation_df (pd.DataFrame): DataFrame containing fixation results.
    - saccade_df (pd.DataFrame): DataFrame containing saccade results.
    """
    logger.info(f"Starting detection for {len(nan_removed_gaze_data_df)} runs.")
    # Collect all rows to process for both agents
    df_keys_for_tasks = nan_removed_gaze_data_df[['session_name', 'interaction_type', 'run_number', 'agent', 'positions']].values.tolist()
    use_one_run = params.get('try_using_single_run')
    if use_one_run:
        df_keys_for_tasks = [df_keys_for_tasks[3]]
        logger.warning(f"!! Testing using positions data from single run: {df_keys_for_tasks[0]}!!")
    # Save params for HPC job submission if needed
    params_file_path = os.path.join(params['processed_data_dir'], 'params.pkl')
    with open(params_file_path, 'wb') as f:
        pickle.dump(params, f)
    logger.info(f"Pickle dumped params to {params_file_path}")
    fixation_rows = []
    saccade_rows = []
    if params.get('recompute_fix_and_saccades_through_hpc_jobs', False):
        # Use HPCFixAndSaccadeDetector to submit jobs for each task
        detector = hpc_fix_and_saccade_detector.HPCFixAndSaccadeDetector(params)
        job_file_path = detector.generate_job_file(df_keys_for_tasks, params_file_path)
        detector.submit_job_array(job_file_path)
        # Combine results after jobs have completed
        hpc_data_subfolder = params.get('hpc_job_output_subfolder', '')
        for task in df_keys_for_tasks:
            session, interaction_type, run, agent, _ = task
            logger.info(f'Updating fix/saccade results for: {session}, {interaction_type}, {run}, {agent}')
            run_str = str(run)
            fix_path = os.path.join(
                params['processed_data_dir'],
                hpc_data_subfolder,
                f'fixation_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            sacc_path = os.path.join(
                params['processed_data_dir'],
                hpc_data_subfolder,
                f'saccade_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            if os.path.exists(fix_path):
                with open(fix_path, 'rb') as f:
                    fix_indices = pickle.load(f)
                    fixation_rows.append({
                        'session_name': session,
                        'interaction_type': interaction_type,
                        'run_number': run,
                        'agent': agent,
                        'fixation_start_stop': fix_indices
                    })
            if os.path.exists(sacc_path):
                with open(sacc_path, 'rb') as f:
                    sacc_indices = pickle.load(f)
                    saccade_rows.append({
                        'session_name': session,
                        'interaction_type': interaction_type,
                        'run_number': run,
                        'agent': agent,
                        'saccade_start_stop': sacc_indices
                    })
    else:
        # Run in serial mode
        logger.info("Running in serial mode.")
        for task in tqdm(df_keys_for_tasks, total=len(df_keys_for_tasks), desc="Processing runs"):
            session, interaction_type, run, agent, positions = task
            if positions is not None and positions.size > 0:
                fix_indices, sacc_indices = process_fix_and_saccade_for_specific_run(
                    session, positions, params
                )
                # Store fixation results
                fixation_rows.append({
                    'session_name': session,
                    'interaction_type': interaction_type,
                    'run_number': run,
                    'agent': agent,
                    'fixation_start_stop': fix_indices
                })
                # Store saccade results
                saccade_rows.append({
                    'session_name': session,
                    'interaction_type': interaction_type,
                    'run_number': run,
                    'agent': agent,
                    'saccade_start_stop': sacc_indices
                })
    # Convert fixation and saccade lists to DataFrames
    fixation_df = pd.DataFrame(fixation_rows)
    saccade_df = pd.DataFrame(saccade_rows)
    logger.info("Detection completed for both agents.")
    return fixation_df, saccade_df


def process_fix_and_saccade_for_specific_run(session_name, positions, params):
    """
    Detects fixations and saccades for a specific run and handles NaNs in the data.
    Parameters:
    - session_name (str): The session identifier.
    - positions (np.ndarray): The position data for the run (Nx2 array).
    - params (dict): Configuration parameters.
    Returns:
    - fixation_start_stop (np.ndarray): Nx2 array of start-stop indices for fixations.
    - saccade_start_stop (np.ndarray): Nx2 array of start-stop indices for saccades.
    """
    # Initialize detectors
    fixation_detector = fixation_detector_class.FixationDetector(
        session_name=session_name,
        samprate=params.get('sampling_rate', 1 / 1000),
        params=params,
        num_cpus=params.get('num_cpus', 1)
    )
    saccade_detector = saccade_detector_class.SaccadeDetector(
        session_name=session_name,
        samprate=params.get('sampling_rate', 1 / 1000),
        params=params,
        num_cpus=params.get('num_cpus', 1)
    )

    # Preprocess positions
    positions = _interpolate_single_nans(positions)
    chunks, offsets = _split_on_nan_islands(positions)

    # Detect fixations and saccades in chunks
    fixation_results, saccade_results = [], []
    for i, (chunk_index, chunk) in enumerate(chunks):
        fixation_res = fixation_detector.detect_fixations((chunk[:, 0], chunk[:, 1]))
        saccade_res = saccade_detector.detect_saccades_with_edge_outliers((chunk[:, 0], chunk[:, 1]))
        # Offset the indices
        chunk_offset = offsets[chunk_index] if chunk_index < len(offsets) else 0
        if 'fixationindices' in fixation_res and len(fixation_res['fixationindices']) > 0:
            fixation_indices = np.atleast_2d(np.array(fixation_res['fixationindices']))
            if fixation_indices.size > 0 and fixation_indices.shape[1] == 2:
                offset_fixation_indices = _offset_indices(fixation_indices, chunk_offset)
                fixation_results.append(offset_fixation_indices)
        if 'saccadeindices' in saccade_res and len(saccade_res['saccadeindices']) > 0:
            saccade_indices = np.atleast_2d(np.array(saccade_res['saccadeindices']))
            if saccade_indices.size > 0 and saccade_indices.shape[1] == 2:
                offset_saccade_indices = _offset_indices(saccade_indices, chunk_offset)
                saccade_results.append(offset_saccade_indices)
    # Combine results
    combined_fixation_results = np.vstack(fixation_results) if fixation_results else np.array([], dtype=np.int64).reshape(0, 2)
    combined_saccade_results = np.vstack(saccade_results) if saccade_results else np.array([], dtype=np.int64).reshape(0, 2)
    # Validate indices
    max_index = len(positions)
    if np.any(combined_fixation_results >= max_index):
        raise ValueError(f"Fixation indices exceed positions array length: {max_index}.")
    if np.any(combined_saccade_results >= max_index):
        raise ValueError(f"Saccade indices exceed positions array length: {max_index}.")
    return combined_fixation_results, combined_saccade_results



def _interpolate_single_nans(data):
    """
    Interpolate single NaN values in the position data.
    Parameters:
    - data (np.ndarray): Nx2 array of position data.
    Returns:
    - np.ndarray: Position data with single NaNs interpolated.
    """
    x, y = data[:, 0], data[:, 1]
    nan_mask = np.isnan(x) | np.isnan(y)
    if np.any(nan_mask):
        single_nan_mask = nan_mask & ~np.isnan(np.roll(x, 1)) & ~np.isnan(np.roll(x, -1))
        for dim, array in enumerate([x, y]):
            if np.any(single_nan_mask):
                valid_idx = ~nan_mask
                interp_func = interp1d(
                    np.flatnonzero(valid_idx), array[valid_idx], kind='linear', bounds_error=False, fill_value="extrapolate"
                )
                array[single_nan_mask] = interp_func(np.flatnonzero(single_nan_mask))
    return np.column_stack((x, y))


def _split_on_nan_islands(data):
    """
    Split data into chunks based on islands of NaNs.
    Returns:
    - List of valid chunks (index, chunk).
    - Array of offsets corresponding to the start of each chunk in the original data.
    """
    nan_mask = np.isnan(data[:, 0]) | np.isnan(data[:, 1])
    split_indices = np.flatnonzero(np.diff(nan_mask.astype(int))) + 1
    chunks = np.split(data, split_indices)
    # Calculate offsets based on the start index of each chunk
    chunk_start_indices = np.insert(split_indices, 0, 0)
    offsets = chunk_start_indices[:-1]
    # Ensure chunks only contain valid data
    valid_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if not np.any(np.isnan(chunk))]
    return valid_chunks, offsets



def _offset_indices(indices, offset):
    """
    Offset detected indices by a given offset.
    Parameters:
    - indices (np.ndarray or list): Indices to offset.
    - offset (int): Offset value.
    Returns:
    - np.ndarray: Offset indices.
    """
    if isinstance(indices, list) and len(indices) == 0:
        return np.array([])
    return indices + offset



























def collect_paths_to_pos_in_dict(gaze_data_dict):
    """
    Collects all paths within the nested gaze_data_dict where 
    the fixation and saccade detectors need to be executed.
    Parameters:
    - gaze_data_dict (dict): The pruned gaze data dictionary with NaN values removed.
    Returns:
    - paths (list): A list of tuples, each containing (session, interaction_type, run, agent).
    """
    paths = []
    for session, session_dict in gaze_data_dict.items():
        for interaction_type, interaction_dict in session_dict.items():
            for run, run_data in interaction_dict.items():
                # Check if positions for m1 and m2 exist and add them to the paths
                if 'm1' in run_data.get('positions', {}):
                    paths.append((session, interaction_type, run, 'm1'))
                    logger.debug(f"Collected path for session: {session}, interaction_type: {interaction_type}, run: {run}, agent: m1")
                if 'm2' in run_data.get('positions', {}):
                    paths.append((session, interaction_type, run, 'm2'))
                    logger.debug(f"Collected path for session: {session}, interaction_type: {interaction_type}, run: {run}, agent: m2")
    logger.info(f"Collected {len(paths)} paths for both agents.")
    return paths



