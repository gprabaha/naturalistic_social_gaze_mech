#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:34:44 2024

@author: pg496
"""

import logging
import random
import fixation_detector_class
import saccade_detector_class
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle

from hpc_fix_and_saccade_detector import HPCFixAndSaccadeDetector


logger = logging.getLogger(__name__)

def detect_fixations_and_saccades(nan_removed_gaze_data_dict, params):
    fixation_dict = {}
    saccade_dict = {}
    # Collect all paths to process for both agents
    paths = collect_paths_to_pos_in_dict(nan_removed_gaze_data_dict)
    logger.info(f"Starting detection. Total runs to process: {len(paths)} for both agents.")
    # Optionally limit to a randomly chosen single example run if specified in params
    if params.get('try_using_single_run', False) and paths:
        paths = [random.choice(paths)]  # Choose a random path
        logger.info("Selected a single run for processing based on params setting.")
    # Prepare tasks with session, interaction_type, run, and agent
    dict_entries_for_tasks = [(session, interaction_type, run, agent) for session, interaction_type, run, agent in paths]
    # Save params for HPC job submission
    params_file_path = os.path.join(params['processed_data_dir'], 'params.pkl')
    with open(params_file_path, 'wb') as f:
        pickle.dump(params, f)
    logger.info(f"Pickle dumped params to {params_file_path}")
    if params.get('use_parallel', False):
        if params.get('recompute_fix_and_saccades_through_hpc_jobs', False):
            # Use HPCFixAndSaccadeDetector to submit jobs for each task
            detector = HPCFixAndSaccadeDetector(params)
            job_file_path = detector.generate_job_file(dict_entries_for_tasks, params_file_path)
            detector.submit_job_array(job_file_path)
        # Combine results after jobs have completed
        hpc_data_subfolder = params.get('hpc_job_output_subfolder', '')
        for task in dict_entries_for_tasks:
            session, interaction_type, run, agent = task
            logger.info(f'Updating fix/sacc dict for: {session}, {interaction_type}, {str(run)}, {agent}')
            run_str = str(run)  # Convert run to string for file path
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
                    fixation_dict.setdefault(session, {}).setdefault(interaction_type, {}).setdefault(run, {})[agent] = fix_dict[session][interaction_type][run][agent]
            if os.path.exists(sacc_path):
                with open(sacc_path, 'rb') as f:
                    sacc_dict = pickle.load(f)
                    saccade_dict.setdefault(session, {}).setdefault(interaction_type, {}).setdefault(run, {})[agent] = sacc_dict[session][interaction_type][run][agent]
    else:
        # Run in serial mode
        logger.info("Running in serial mode.")
        results = [process_fix_and_saccade_for_specific_run(
            task, nan_removed_gaze_data_dict) for task in tqdm(
                dict_entries_for_tasks,
                total=len(dict_entries_for_tasks),
                desc="Processing runs")]
        # Combine results
        for fix_dict, sacc_dict in results:
            for session, interaction_types in fix_dict.items():
                for interaction_type, runs in interaction_types.items():
                    for run, agents in runs.items():
                        fixation_dict.setdefault(session, {}).setdefault(interaction_type, {}).setdefault(run, {}).update(agents)
            for session, interaction_types in sacc_dict.items():
                for interaction_type, runs in interaction_types.items():
                    for run, agents in runs.items():
                        saccade_dict.setdefault(session, {}).setdefault(interaction_type, {}).setdefault(run, {}).update(agents)
    logger.info("Detection completed for both agents.")
    # Save the fixation and saccade dictionaries to the processed data directory
    try:
        fixation_output_path = os.path.join(params['processed_data_dir'], 'fixation_dict.pkl')
        saccade_output_path = os.path.join(params['processed_data_dir'], 'saccade_dict.pkl')
        with open(fixation_output_path, 'wb') as f:
            pickle.dump(fixation_dict, f)
        logger.info(f"Fixation dictionary saved successfully at {fixation_output_path}")
        with open(saccade_output_path, 'wb') as f:
            pickle.dump(saccade_dict, f)
        logger.info(f"Saccade dictionary saved successfully at {saccade_output_path}")
    except Exception as e:
        logger.error(f"Failed to save fixation and saccade dictionaries: {e}")
    return fixation_dict, saccade_dict


def process_fix_and_saccade_for_specific_run(args, nan_removed_gaze_data_dict):
    """
    Processes fixations and saccades for a specific run.
    Parameters:
    - args (tuple): A tuple containing session, interaction_type, run, agent, and params.
    - gaze_data_dict (dict): The gaze data dictionary containing position data.
    Returns:
    - fixation_dict (dict): A dictionary containing fixation detection results for the specific run.
    - saccade_dict (dict): A dictionary containing saccade detection results for the specific run.
    """
    session, interaction_type, run, agent, params = args
    fixation_dict, saccade_dict = {}, {}
    # Fetch the required positions data from the gaze_data_dict
    positions = nan_removed_gaze_data_dict[session][interaction_type][run].get('positions', {}).get(agent)
    if positions is not None and positions.size > 0:
        # Initialize fixation and saccade detectors
        fixation_detector = fixation_detector_class.FixationDetector(
            session_name=session,
            samprate=params.get('sampling_rate', 1 / 1000),
            params=params,
            num_cpus=params.get('num_cpus', 1)
        )
        saccade_detector = saccade_detector_class.SaccadeDetector(
            session_name=session,
            samprate=params.get('sampling_rate', 1 / 1000),
            params=params,
            num_cpus=params.get('num_cpus', 1)
        )
        # Detect fixations
        fixation_results = fixation_detector.detect_fixations_with_edge_outliers(
            (positions[0], positions[1])
        )
        fixation_dict[session] = {interaction_type: {run: {agent: fixation_results}}}
        logger.info(f"Detected fixations for session: {session}, interaction_type: {interaction_type}, run: {run}, , agent: {agent}")
        # Detect saccades
        saccade_results = saccade_detector.detect_saccades_with_edge_outliers(
            (positions[0], positions[1])
        )
        saccade_dict[session] = {interaction_type: {run: {agent: saccade_results}}}
        logger.info(f"Detected saccades for session: {session}, interaction_type: {interaction_type}, run: {run}, agent: {agent}")
    return fixation_dict, saccade_dict


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



