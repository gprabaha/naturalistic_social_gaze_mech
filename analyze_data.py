import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
import os
from scipy.signal import fftconvolve
import logging

import util

import pdb


# Set up a logger for this script
logger = logging.getLogger(__name__)


def create_binary_timeline_for_behavior(behavior_df, nan_removed_gaze_data_df, behavior_type='fixation'):
    """
    Creates a binary timeline DataFrame indicating the presence of specific behaviors
    (fixations or saccades) across a timeline for each session, interaction, run, and agent.
    Parameters:
    - behavior_df (pd.DataFrame): DataFrame containing behavior data (fixations or saccades).
    - nan_removed_gaze_data_df (pd.DataFrame): DataFrame with neural timeline data per session.
    - behavior_type (str): Type of behavior to process, either 'fixation' or 'saccade'.
    Returns:
    - binary_timeline_df (pd.DataFrame): DataFrame with binary timelines for each behavior event.
    """
    # Ensure behavior_type is valid
    if behavior_type not in ['fixation', 'saccade']:
        raise ValueError("behavior_type must be 'fixation' or 'saccade'")
    # Parallel processing using a separate function to generate row data
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(_generate_binary_timeline_for_df_row, [
            (index, row, nan_removed_gaze_data_df, behavior_type) for index, row in behavior_df.iterrows()
        ]),
        total=len(behavior_df),
        desc=f"Generating binary timeline for {behavior_type}"))
    # Flatten and concatenate the list of DataFrames, then sort by index for order
    binary_timeline_df = pd.concat(results).sort_index().reset_index(drop=True)
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
    # Use the appropriate helper function for fixation or saccade and return the result as a DataFrame
    if behavior_type == 'fixation':
        result = __append_fixation_data(behav_row, total_timeline_length)
    elif behavior_type == 'saccade':
        result = __append_saccade_data(behav_row, total_timeline_length)
    # Return DataFrame with index for sorting
    return pd.DataFrame(result).assign(row_index=index)


def __append_fixation_data(behav_row, total_timeline_length):
    """
    Generates binary timeline entries for fixation events, with separate entries for each unique
    location and an overall entry for all fixations.
    Parameters:
    - behav_row (pd.Series): Row data from the fixation DataFrame.
    - total_timeline_length (int): Total length of the timeline for this session and run.
    Returns:
    - List[dict]: List of dictionaries with binary timeline data for each fixation.
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
    result_rows = []
    binary_timeline_all = np.zeros(total_timeline_length, dtype=int)
    # Create binary timelines for each unique location
    for (start, stop), loc in zip(intervals, locations):
        binary_timeline_all[start:stop] = 1  # Mark fixation in the "all" timeline
        # Create individual location entries
        location_timeline = np.zeros(total_timeline_length, dtype=int)
        location_timeline[start:stop] = 1
        result_rows.append({
            'session_name': session,
            'interaction_type': interaction,
            'run_number': run,
            'agent': agent,
            'behav_type': behavior_type,
            'from': loc,
            'to': loc,
            'binary_timeline': location_timeline.tolist()
        })
    # Add an "all fixations" entry
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
    from-to location and an overall entry for all saccades.
    Parameters:
    - behav_row (pd.Series): Row data from the saccade DataFrame.
    - total_timeline_length (int): Total length of the timeline for this session and run.
    Returns:
    - List[dict]: List of dictionaries with binary timeline data for each saccade.
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
    result_rows = []
    binary_timeline_all = np.zeros(total_timeline_length, dtype=int)
    # Create binary timelines for each unique from-to combination
    for (start, stop), from_loc, to_loc in zip(intervals, from_locations, to_locations):
        binary_timeline_all[start:stop] = 1  # Mark saccade in the "all" timeline
        # Create individual from-to entries
        from_to_timeline = np.zeros(total_timeline_length, dtype=int)
        from_to_timeline[start:stop] = 1
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


























# Function to create binary behavior timeseries DataFrame
def create_binary_behav_timeseries_df(fixation_df, saccade_df, gaze_data_df, params):
    """
    Creates the binary behavior timeseries DataFrame by adding fixation and saccade binary vectors.
    Parameters:
    fixation_df (pd.DataFrame): DataFrame containing fixation events.
    saccade_df (pd.DataFrame): DataFrame containing saccade events.
    gaze_data_df (pd.DataFrame): The gaze data DataFrame with NaN removed.
    params (dict): Dictionary containing configuration, including 'use_parallel' and 'num_cpus'.
    Returns:
    pd.DataFrame: The binary behavior timeseries DataFrame.
    """
    logger.info("Starting creation of binary behavior timeseries DataFrame.")
    binary_behav_df = util.initiate_behav_df_label_cols(gaze_data_df)
    logger.info("Adding fixation binary vectors to DataFrame.")
    binary_behav_df = add_bin_vectors_to_behav_df(
        binary_behav_df,
        fixation_df,
        gaze_data_df,
        event_type='fixation',
        use_parallel=params['use_parallel'],
        num_cpus=params['num_cpus']
    )
    logger.info("Adding saccade binary vectors to DataFrame.")
    binary_behav_df = add_bin_vectors_to_behav_df(
        binary_behav_df,
        saccade_df,
        gaze_data_df,
        event_type='saccade',
        use_parallel=params['use_parallel'],
        num_cpus=params['num_cpus']
    )
    logger.info("Finished creation of binary behavior timeseries DataFrame.")
    return binary_behav_df


def add_bin_vectors_to_behav_df(behav_df, event_df, nan_removed_gaze_data_df, event_type, use_parallel=False, num_cpus=4):
    """
    Adds binary vectors (fixation or saccade) to the behavioral dataframe.
    Parameters:
    -----------
    behav_df : pd.DataFrame - Behavioral data for each session, interaction type, agent, and run.
    event_df : pd.DataFrame - Fixation or saccade start-stop intervals for each session, interaction type, agent, and run.
    nan_removed_gaze_data_df : pd.DataFrame - Gaze data with `neural_timeline` column.
    event_type : str - Type of event, 'fixation' or 'saccade'.
    use_parallel : bool - Whether to use parallel processing (default: False).
    num_cpus : int - Number of CPUs to use in parallel processing (default: 4).
    Returns:
    --------
    pd.DataFrame - `behav_df` with an added column `<event_type>_binary_vector` for each run.
    """
    logger.info(f"Adding {event_type} binary vectors to the behavioral DataFrame.")
    event_col_name = f'{event_type}_binary_vector'
    behav_df = behav_df.copy()  # Avoid SettingWithCopyWarning by working on a copy of the DataFrame
    behav_df.loc[:, event_col_name] = None  # Explicitly assign a new column
    # Log whether using parallel or serial processing
    if use_parallel:
        logger.info(f"Using parallel processing with {num_cpus} CPUs.")
    else:
        logger.info("Using serial processing.")
    args = [(idx, row, event_df, nan_removed_gaze_data_df, event_type) for idx, row in behav_df.iterrows()]
    for idx, result in tqdm(map(make_binary_vector_for_run, args), total=len(args), desc="Making binary vectors"):
        if result is not None:
            behav_df.at[idx, event_col_name] = result  # Update with .loc[] to avoid the warning
    logger.info(f"Finished adding {event_type} binary vectors.")
    return behav_df


def make_binary_vector_for_run(args):
    """
    Creates a binary vector for a specific run based on event start-stop intervals.
    Parameters:
    -----------
    args : tuple - Contains index, `behav_df` row, event dataframe, gaze data, and event type.
    Returns:
    --------
    tuple - (idx, binary_vector) where idx is the row index and binary_vector is the created binary vector.
    """
    idx, row, event_df, nan_removed_gaze_data_df, event_type = args
    session, interaction, run, agent = row['session_name'], row['interaction_type'], row['run_number'], row['agent']
    logger.debug(f"Processing {event_type} binary vector for session {session}, interaction {interaction}, run {run}, agent {agent}.")
    event_start_stop_col = f'{event_type}_start_stop'
    event_row = event_df[
        (event_df['session_name'] == session) &
        (event_df['interaction_type'] == interaction) &
        (event_df['run_number'] == run) &
        (event_df['agent'] == agent)
    ]
    if not event_row.empty:
        event_intervals = np.array(event_row[event_start_stop_col].values[0])
        neural_timeline = nan_removed_gaze_data_df[
            (nan_removed_gaze_data_df['session_name'] == session) &
            (nan_removed_gaze_data_df['interaction_type'] == interaction) &
            (nan_removed_gaze_data_df['run_number'] == run) &
            (nan_removed_gaze_data_df['agent'] == agent)
        ]['neural_timeline'].values[0]
        if event_intervals.size > 0:
            logger.debug(f"Found {len(event_intervals)} event intervals for session {session}, run {run}.")
            binary_vector = np.zeros(len(neural_timeline), dtype=int)
            index_ranges = np.concatenate([np.arange(start, stop + 1) for start, stop in event_intervals]).astype(int)
            binary_vector[index_ranges] = 1
            return idx, binary_vector
    return idx, None


def compute_scaled_autocorrelations_for_behavior_df(df, params):
    """
    Computes the scaled autocorrelations for fixation and saccade binary vectors across all rows of the given DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing binary vectors for fixation and saccade events.
    params (dict): Dictionary containing configuration, including 'num_cpus', 'use_parallel', and 'processed_data_dir'.
    Returns:
    pd.DataFrame: A new DataFrame containing the autocorrelations for fixation and saccade vectors.
    """
    logger.info("Starting computation of scaled autocorrelations for the behavior DataFrame.")
    num_cpus = params['num_cpus']
    use_parallel = params['use_parallel']
    processed_data_dir = params['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)
    if use_parallel:
        logger.info(f"Running autocorrelations in parallel using {num_cpus} CPUs.")
        results = Parallel(n_jobs=num_cpus)(
            delayed(_compute_autocorr_for_row)(row, index) for index, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Making autocorrelations for df rows (Parallel)")
        )
    else:
        logger.info("Running autocorrelations serially.")
        results = [_compute_autocorr_for_row(row, index) for index, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Making autocorrelations for df rows (Serial)")
        ]
    logger.info("Creating DataFrame to store the results.")
    autocorr_df = pd.DataFrame()
    autocorr_df['session_name'] = df['session_name']
    autocorr_df['interaction_type'] = df['interaction_type']
    autocorr_df['run_number'] = df['run_number']
    autocorr_df['agent'] = df['agent']
    autocorr_df['fixation_scaled_autocorr'] = [None] * len(df)
    autocorr_df['saccade_scaled_autocorr'] = [None] * len(df)
    for result in results:
        index, fixation_autocorr, saccade_autocorr = result
        autocorr_df.at[index, 'fixation_scaled_autocorr'] = fixation_autocorr
        autocorr_df.at[index, 'saccade_scaled_autocorr'] = saccade_autocorr
    logger.info("Finished computation of scaled autocorrelations.")
    return autocorr_df


def _compute_autocorr_for_row(row, index):
    """
    Computes scaled autocorrelations for both fixation and saccade binary vectors for a given row.
    Parameters:
    row (pd.Series): A single row from the DataFrame.
    index (int): The index of the row in the DataFrame.
    Returns:
    tuple: A tuple containing the index and scaled autocorrelations for fixation and saccade binary vectors.
    """
    logger.debug(f"Computing autocorrelations for row {index}.")
    fixation_autocorr = _compute_scaled_autocorr_fft(row['fixation_binary_vector'])
    saccade_autocorr = _compute_scaled_autocorr_fft(row['saccade_binary_vector'])
    return index, fixation_autocorr, saccade_autocorr


def _compute_scaled_autocorr_fft(binary_vector):
    """
    Computes the scaled autocorrelation for a given binary vector using FFT for faster computation.
    Parameters:
    binary_vector (list): A binary vector representing either fixation or saccade events.
    Returns:
    list: A list of scaled autocorrelation values.
    """
    n = len(binary_vector)
    logger.debug(f"Computing FFT-based autocorrelation for binary vector of length {n}.")
    autocorr_full = fftconvolve(binary_vector, binary_vector[::-1], mode='full')
    autocorrs = autocorr_full[n-1:]
    valid_lengths = np.arange(n, 0, -1)
    scaled_autocorrs = autocorrs / valid_lengths
    return scaled_autocorrs


