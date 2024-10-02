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

import hpc_fix_and_saccade_detector
import fixation_detector_class
import saccade_detector_class


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
            detector = hpc_fix_and_saccade_detector.HPCFixAndSaccadeDetector(params)
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



