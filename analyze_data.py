import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import os
from scipy.signal import fftconvolve
import logging

import util

import pdb


# Set up a logger for this script
logger = logging.getLogger(__name__)

def create_binary_timeline_for_behavior(behavior_df, nan_removed_gaze_data_df, num_cpus, behavior_type='fixation', use_parallel=False):
    """
    Creates a binary timeline DataFrame indicating the presence of specific behaviors
    (fixations or saccades) across a timeline for each session, interaction, run, and agent.
    Parameters:
    - behavior_df (pd.DataFrame): DataFrame containing behavior data (fixations or saccades).
    - nan_removed_gaze_data_df (pd.DataFrame): DataFrame with neural timeline data per session.
    - num_cpus (int): Number of CPUs for parallel processing.
    - behavior_type (str): Type of behavior to process, either 'fixation' or 'saccade'.
    - use_parallel (bool): Flag indicating whether to use parallel processing.
    Returns:
    - binary_timeline_df (pd.DataFrame): DataFrame with binary timelines for each behavior event.
    """
    if behavior_type not in ['fixation', 'saccade']:
        raise ValueError("behavior_type must be 'fixation' or 'saccade'")
    if use_parallel:
        results = Parallel(n_jobs=num_cpus)(
            delayed(_generate_binary_timeline_for_df_row)((index, row, nan_removed_gaze_data_df, behavior_type))
            for index, row in tqdm(behavior_df.iterrows(), total=len(behavior_df), desc=f"Generating binary timeline for {behavior_type}")
        )
    else:
        results = [
            _generate_binary_timeline_for_df_row((index, row, nan_removed_gaze_data_df, behavior_type))
            for index, row in tqdm(behavior_df.iterrows(), total=len(behavior_df), desc=f"Generating binary timeline for {behavior_type}")
        ]
    # Concatenate all results and reset the index
    binary_timeline_df = pd.concat(results).sort_values(by='row_index_in_behav_df').reset_index(drop=True)
    return binary_timeline_df


def _generate_binary_timeline_for_df_row(args):
    """
    Helper function to process each row of the behavior DataFrame and generate binary timelines.
    Parameters:
    - args (tuple): Tuple containing row index, row data, neural timeline DataFrame, and behavior type.
    Returns:
    - pd.DataFrame: DataFrame with binary timelines for each event in the row.
    """
    index, behav_row, nan_removed_gaze_data_df, behavior_type = args
    session = behav_row['session_name']
    interaction = behav_row['interaction_type']
    run = behav_row['run_number']
    agent = behav_row['agent']
    # Fetch the neural timeline for the current session, run, and agent
    neural_timeline_row = nan_removed_gaze_data_df[
        (nan_removed_gaze_data_df['session_name'] == session) &
        (nan_removed_gaze_data_df['interaction_type'] == interaction) &
        (nan_removed_gaze_data_df['run_number'] == run) &
        (nan_removed_gaze_data_df['agent'] == agent)
    ]
    if neural_timeline_row.empty:
        return pd.DataFrame()  # Return empty DataFrame if no matching neural timeline row is found
    total_timeline_length = len(neural_timeline_row.iloc[0]['neural_timeline'])
    # Generate timelines based on the behavior type
    if behavior_type == 'fixation':
        result = __append_fixation_data(behav_row, total_timeline_length)
    elif behavior_type == 'saccade':
        result = __append_saccade_data(behav_row, total_timeline_length)
    # Assign row_index to each row within the result for accurate ordering
    for row in result:
        row['row_index_in_behav_df'] = index
    return pd.DataFrame(result)


def __append_fixation_data(behav_row, total_timeline_length):
    """
    Generates binary timeline entries for fixation events, with separate entries for each unique
    location type and an overall entry for all fixations.
    Parameters:
    - behav_row (pd.Series): Row data from the fixation DataFrame.
    - total_timeline_length (int): Total length of the timeline for this session and run.
    Returns:
    - List[dict]: List of dictionaries with binary timeline data for each unique location and "all" fixations.
    """
    start_stop_column = 'fixation_start_stop'
    location_column = 'location'
    intervals = behav_row[start_stop_column]
    locations = behav_row[location_column]
    session = behav_row['session_name']
    interaction = behav_row['interaction_type']
    run = behav_row['run_number']
    agent = behav_row['agent']
    behavior_type = 'fixation'
    # Initialize the "all" binary timeline
    binary_timeline_all = np.zeros(total_timeline_length, dtype=int)
    # Create a dictionary to store binary timelines for each unique location
    location_timelines = {loc: np.zeros(total_timeline_length, dtype=int) for loc in set(locations)}
    # Update binary timelines based on intervals for each location
    for (start, stop), loc in zip(intervals, locations):
        binary_timeline_all[start:stop] = 1  # Mark interval in the "all" timeline
        location_timelines[loc][start:stop] = 1  # Mark interval for the specific location timeline
    # Prepare results for each unique location type
    result_rows = []
    for loc, loc_timeline in location_timelines.items():
        result_rows.append({
            'session_name': session,
            'interaction_type': interaction,
            'run_number': run,
            'agent': agent,
            'behav_type': behavior_type,
            'from': loc,
            'to': loc,
            'binary_timeline': loc_timeline.tolist()
        })
    # Add the "all fixations" entry
    result_rows.append({
        'session_name': session,
        'interaction_type': interaction,
        'run_number': run,
        'agent': agent,
        'behav_type': behavior_type,
        'from': 'all',
        'to': 'all',
        'binary_timeline': binary_timeline_all.tolist()
    })
    return result_rows


def __append_saccade_data(behav_row, total_timeline_length):
    """
    Generates binary timeline entries for saccade events, with separate entries for each unique
    from-to location combination and an overall entry for all saccades.
    Parameters:
    - behav_row (pd.Series): Row data from the saccade DataFrame.
    - total_timeline_length (int): Total length of the timeline for this session and run.
    Returns:
    - List[dict]: List of dictionaries with binary timeline data for each unique from-to location combination and "all" saccades.
    """
    start_stop_column = 'saccade_start_stop'
    from_column = 'from'
    to_column = 'to'
    intervals = behav_row[start_stop_column]
    from_locations = behav_row[from_column]
    to_locations = behav_row[to_column]
    session = behav_row['session_name']
    interaction = behav_row['interaction_type']
    run = behav_row['run_number']
    agent = behav_row['agent']
    behavior_type = 'saccade'
    # Initialize the "all" binary timeline
    binary_timeline_all = np.zeros(total_timeline_length, dtype=int)
    # Find all unique (from, to) combinations
    unique_combinations = set(zip(from_locations, to_locations))
    # Create a dictionary to store binary timelines for each unique (from, to) combination
    from_to_timelines = { (from_loc, to_loc): np.zeros(total_timeline_length, dtype=int) for from_loc, to_loc in unique_combinations }
    # Update binary timelines based on intervals for each unique (from, to) combination
    for (start, stop), from_loc, to_loc in zip(intervals, from_locations, to_locations):
        binary_timeline_all[start:stop] = 1  # Mark interval in the "all" timeline
        from_to_timelines[(from_loc, to_loc)][start:stop] = 1  # Mark interval for the specific from-to combination
    # Prepare results for each unique (from, to) combination
    result_rows = []
    for (from_loc, to_loc), from_to_timeline in from_to_timelines.items():
        result_rows.append({
            'session_name': session,
            'interaction_type': interaction,
            'run_number': run,
            'agent': agent,
            'behav_type': behavior_type,
            'from': from_loc,
            'to': to_loc,
            'binary_timeline': from_to_timeline.tolist()
        })
    # Add an "all saccades" entry
    result_rows.append({
        'session_name': session,
        'interaction_type': interaction,
        'run_number': run,
        'agent': agent,
        'behav_type': behavior_type,
        'from': 'all',
        'to': 'all',
        'binary_timeline': binary_timeline_all.tolist()
    })
    return result_rows


def calculate_auto_and_cross_corrs_bet_behav_vectors(binary_behav_timeseries_df, num_cpus=1, use_parallel=False):
    """
    Calculates autocorrelations and cross-correlations using FFT for each unique combination
    of behavior type, from, and to within each session, interaction, and run.
    Parameters:
    - binary_behav_timeseries_df (pd.DataFrame): DataFrame with binary behavior timeseries.
    - num_cpus (int): Number of CPUs for parallel processing.
    - use_parallel (bool): Flag to indicate parallel processing.
    Returns:
    - pd.DataFrame: DataFrame with calculated autocorrelations and cross-correlations.
    """
    # Group by session-related keys and process each session group
    run_groups = binary_behav_timeseries_df.groupby(['session_name', 'interaction_type', 'run_number'])
    if use_parallel:
        # Parallel processing with joblib
        results = Parallel(n_jobs=num_cpus)(
            delayed(_compute_session_correlations)(group) for group in tqdm(run_groups, desc="Processing correlations for specific run (parallel)", total=len(run_groups))
        )
    else:
        # Serial processing with tqdm
        results = [
            _compute_session_correlations(group) for group in tqdm(run_groups, desc="Processing correlations for specific run (serial)", total=len(run_groups))
        ]
    # Flatten the list of results and convert to DataFrame
    flat_results = [item for sublist in results for item in sublist]
    correlation_df = pd.DataFrame(flat_results)
    return correlation_df


def _compute_session_correlations(group):
    """
    Process each session group to calculate autocorrelations and cross-correlations.
    Parameters:
    - group (tuple): Tuple containing session information and the corresponding DataFrame group.
    Returns:
    - List[dict]: List of dictionaries with calculated autocorrelations and cross-correlations for each behavior.
    """
    (session_name, interaction_type, run_number), run_df = group
    results = []
    behavior_groups = run_df.groupby(['behav_type', 'from', 'to'])
    for (behav_type, from_loc, to_loc), behavior_df in behavior_groups:
        # Retrieve binary timelines for m1 and m2 agents if available
        agent_timelines = {agent: behavior_df[behavior_df['agent'] == agent]['binary_timeline'].iloc[0]
                           for agent in ['m1', 'm2'] if agent in behavior_df['agent'].values}
        if 'm1' in agent_timelines and 'm2' in agent_timelines:
            binary_timeline_m1 = np.array(agent_timelines['m1'])
            binary_timeline_m2 = np.array(agent_timelines['m2'])
            # Calculate autocorrelations and cross-correlations using FFT
            autocorr_m1 = __fft_autocorr(binary_timeline_m1)
            autocorr_m2 = __fft_autocorr(binary_timeline_m2)
            crosscorr_m1_m2 = __fft_crosscorr(binary_timeline_m1, binary_timeline_m2)
            crosscorr_m2_m1 = __fft_crosscorr(binary_timeline_m2, binary_timeline_m1)
            # Append results
            results.append({
                'session_name': session_name,
                'interaction_type': interaction_type,
                'run_number': run_number,
                'behav_type': behav_type,
                'from': from_loc,
                'to': to_loc,
                'autocorr_m1': autocorr_m1,
                'autocorr_m2': autocorr_m2,
                'crosscorr_m1_m2': crosscorr_m1_m2,
                'crosscorr_m2_m1': crosscorr_m2_m1
            })
    return results


def __fft_autocorr(binary_vector):
    """Compute and scale autocorrelation using FFT."""
    n = len(binary_vector)
    autocorr_full = fftconvolve(binary_vector, binary_vector[::-1], mode='full')
    autocorrs = autocorr_full[n-1:]
    valid_lengths = np.arange(n, 0, -1)
    scaled_autocorrs = autocorrs / valid_lengths
    return scaled_autocorrs


def __fft_crosscorr(binary_vector_x, binary_vector_y):
    """Compute and scale cross-correlation using FFT."""
    n = len(binary_vector_x)
    crosscorr_full = fftconvolve(binary_vector_x, binary_vector_y[::-1], mode='full')
    crosscorrs = crosscorr_full[n-1:]
    valid_lengths = np.arange(n, 0, -1)
    scaled_crosscorrs = crosscorrs / valid_lengths
    return scaled_crosscorrs





























