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

import hpc_fix_and_saccade_detector
import fixation_detector_class
import saccade_detector_class


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
    # Save params for HPC job submission if needed
    params_file_path = os.path.join(params['processed_data_dir'], 'params.pkl')
    with open(params_file_path, 'wb') as f:
        pickle.dump(params, f)
    logger.info(f"Pickle dumped params to {params_file_path}")
    fixation_rows = []
    saccade_rows = []
    if params.get('use_parallel', False):
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
                    fix_dict = pickle.load(f)
                    fixation_rows.append({
                        'session_name': session,
                        'interaction_type': interaction_type,
                        'run_number': run,
                        'agent': agent,
                        'fixation_start_stop': fix_dict['fixationindices']
                    })
            if os.path.exists(sacc_path):
                with open(sacc_path, 'rb') as f:
                    sacc_dict = pickle.load(f)
                    saccade_rows.append({
                        'session_name': session,
                        'interaction_type': interaction_type,
                        'run_number': run,
                        'agent': agent,
                        'saccade_start_stop': sacc_dict['saccadeindices']
                    })
    else:
        # Run in serial mode
        logger.info("Running in serial mode.")
        for task in tqdm(df_keys_for_tasks, total=len(df_keys_for_tasks), desc="Processing runs"):
            session, interaction_type, run, agent, positions = task
            if positions is not None and positions.size > 0:
                fix_dict, sacc_dict = process_fix_and_saccade_for_specific_run(
                    session, positions, params
                )
                # Store fixation results
                fixation_rows.append({
                    'session_name': session,
                    'interaction_type': interaction_type,
                    'run_number': run,
                    'agent': agent,
                    'fixation_start_stop': fix_dict['fixationindices']
                })
                # Store saccade results
                saccade_rows.append({
                    'session_name': session,
                    'interaction_type': interaction_type,
                    'run_number': run,
                    'agent': agent,
                    'saccade_start_stop': sacc_dict['saccadeindices']
                })
    # Convert fixation and saccade lists to DataFrames
    fixation_df = pd.DataFrame(fixation_rows)
    saccade_df = pd.DataFrame(saccade_rows)
    logger.info("Detection completed for both agents.")
    return fixation_df, saccade_df


def process_fix_and_saccade_for_specific_run(session_name, positions, params):
    """
    Detects fixations and saccades for a specific run and returns start-stop indices for both.
    Parameters:
    - session_name (str): The session identifier.
    - interaction_type (str): The type of interaction.
    - run_number (int): The run number.
    - agent (str): The agent identifier ('m1' or 'm2').
    - positions (np.ndarray): The position data for the run (Nx2 array).
    - params (dict): Configuration parameters.
    Returns:
    - fixation_start_stop (np.ndarray): Nx2 array of start-stop indices for fixations.
    - saccade_start_stop (np.ndarray): Nx2 array of start-stop indices for saccades.
    """
    # Initialize fixation and saccade detectors
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
    # Detect fixations
    fixation_results = fixation_detector.detect_fixations_with_edge_outliers(
        (positions[:, 0], positions[:, 1])
    )
    # Detect saccades
    saccade_results = saccade_detector.detect_saccades_with_edge_outliers(
        (positions[:, 0], positions[:, 1])
    )
    return fixation_results, saccade_results



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



