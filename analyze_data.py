import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import os
from scipy.signal import fftconvolve
import logging
import glob

from hpc_shuffled_cross_corr import HPCShuffledCrossCorr
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
    # Flatten the list of lists
    flattened_locations = [item for sublist in locations for item in sublist]
    # Get unique entries using set
    unique_locations = list(set(flattened_locations))
    # Initialize the "all" binary timeline
    binary_timeline_all = np.zeros(total_timeline_length, dtype=int)
    # Create a dictionary to store binary timelines for each unique location
    location_timelines = {loc: np.zeros(total_timeline_length, dtype=int) for loc in set(unique_locations)}
    # Update binary timelines based on intervals for each location
    # Update binary timelines based on intervals for each location
    for (start, stop), loc_list in zip(intervals, locations):
        binary_timeline_all[start:stop] = 1  # Mark interval in the "all" timeline
        for loc in loc_list:  # Handle multiple locations
            location_timelines[loc][start:stop] = 1  # Mark interval for each specific location timeline
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
    # result_rows.append({
    #     'session_name': session,
    #     'interaction_type': interaction,
    #     'run_number': run,
    #     'agent': agent,
    #     'behav_type': behavior_type,
    #     'from': 'all',
    #     'to': 'all',
    #     'binary_timeline': binary_timeline_all.tolist()
    # })
    return result_rows


def __append_saccade_data(behav_row, total_timeline_length):
    """
    Generates binary timeline entries for saccade events, with separate entries for each unique
    `from` location and `to` location.
    Parameters:
    - behav_row (pd.Series): Row data from the saccade DataFrame.
    - total_timeline_length (int): Total length of the timeline for this session and run.
    Returns:
    - List[dict]: List of dictionaries with binary timeline data for each unique `from` and `to` location.
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
    # Extract unique `from` and `to` locations
    unique_from_locations = set(loc for loc_list in from_locations for loc in loc_list)
    unique_to_locations = set(loc for loc_list in to_locations for loc in loc_list)
    # Initialize binary timelines for all `from` and `to` locations
    from_timelines = {from_loc: np.zeros(total_timeline_length, dtype=int) for from_loc in unique_from_locations}
    to_timelines = {to_loc: np.zeros(total_timeline_length, dtype=int) for to_loc in unique_to_locations}
    # Update binary timelines based on intervals
    for (start, stop), from_list, to_list in zip(intervals, from_locations, to_locations):
        # Update `from` timelines, labeling `to` as "elsewhere"
        for from_loc in from_list:
            from_timelines[from_loc][start:stop] = 1
        # Update `to` timelines, labeling `from` as "elsewhere"
        for to_loc in to_list:
            to_timelines[to_loc][start:stop] = 1
    # Prepare result rows for `from` locations
    result_rows = []
    for from_loc, from_timeline in from_timelines.items():
        result_rows.append({
            'session_name': session,
            'interaction_type': interaction,
            'run_number': run,
            'agent': agent,
            'behav_type': behavior_type,
            'from': from_loc,
            'to': 'elsewhere',
            'binary_timeline': from_timeline.tolist()
        })
    # Prepare result rows for `to` locations
    for to_loc, to_timeline in to_timelines.items():
        result_rows.append({
            'session_name': session,
            'interaction_type': interaction,
            'run_number': run,
            'agent': agent,
            'behav_type': behavior_type,
            'from': 'elsewhere',
            'to': to_loc,
            'binary_timeline': to_timeline.tolist()
        })
    return result_rows


def merge_left_and_right_object_timelines_in_behav_df(binary_behav_timeseries_df):
    """
    Merge left and right object timelines in the behavioral dataframe:
    - For fixations, merge 'left_nonsocial_object' and 'right_nonsocial_object' into 'object'
      using an OR operation on the binary timelines.
    - For saccades, merge 'left_nonsocial_object' and 'right_nonsocial_object' separately
      for `from` and `to` fields, using an OR operation on the binary timelines.
    Args:
        binary_behav_timeseries_df (pd.DataFrame): Input dataframe containing binary timelines.
    Returns:
        pd.DataFrame: Updated dataframe with merged object timelines.
    """
    logger.info("Starting the merge of left and right object timelines.")
    updated_group_dfs = []  # Store updated group DataFrames
    # Group by event and behavior type
    grouped = binary_behav_timeseries_df.groupby(
        ['session_name', 'interaction_type', 'run_number', 'agent', 'behav_type']
    )
    logger.info(f"Processing {len(grouped)} groups of data.")
    for group_idx, (group_keys, group_df) in enumerate(grouped, start=1):
        session_name, interaction_type, run_number, agent, behav_type = group_keys
        logger.debug(f"Processing group {group_idx}: Session {session_name}, Interaction {interaction_type}, "
                     f"Run {run_number}, Agent {agent}, Behavior Type {behav_type}.")
        updated_group_df = group_df.copy()
        if behav_type == 'fixation':
            updated_group_df = _merge_fixation_object_timelines(updated_group_df, session_name, interaction_type, run_number, agent, behav_type, group_idx)
        elif behav_type == 'saccade':
            updated_group_df = _merge_saccade_object_timelines(updated_group_df, session_name, interaction_type, run_number, agent, behav_type, group_idx)
        updated_group_dfs.append(updated_group_df)
    # Concatenate all updated groups
    updated_df = pd.concat(updated_group_dfs, ignore_index=True)
    logger.info("Merging completed successfully.")
    return updated_df


def _merge_fixation_object_timelines(group_df, session_name, interaction_type, run_number, agent, behav_type, group_idx):
    """
    Helper function to merge fixation timelines for left and right nonsocial objects into 'object'.
    """
    object_mask = group_df['from'].isin(['left_nonsocial_object', 'right_nonsocial_object'])
    object_df = group_df[object_mask]
    group_df = group_df[~object_mask]  # Remove left/right rows
    if not object_df.empty:
        merged_timeline = np.bitwise_or.reduce(np.stack(object_df['binary_timeline'].values))
        group_df = pd.concat([
            group_df,
            pd.DataFrame([{
                'session_name': session_name,
                'interaction_type': interaction_type,
                'run_number': run_number,
                'agent': agent,
                'behav_type': behav_type,
                'from': 'object',
                'to': 'object',
                'binary_timeline': merged_timeline.tolist()
            }])
        ], ignore_index=True)
        logger.debug(f"Merged fixation object timelines for group {group_idx}.")
    return group_df


def _merge_saccade_object_timelines(group_df, session_name, interaction_type, run_number, agent, behav_type, group_idx):
    """
    Helper function to merge saccade timelines for left and right nonsocial objects into 'object'.
    """
    # Merge for the `from` field
    from_mask = group_df['from'].isin(['left_nonsocial_object', 'right_nonsocial_object'])
    from_df = group_df[from_mask]
    group_df = group_df[~from_mask]  # Remove left/right rows
    if not from_df.empty:
        merged_timeline = np.bitwise_or.reduce(np.stack(from_df['binary_timeline'].values))
        group_df = pd.concat([
            group_df,
            pd.DataFrame([{
                'session_name': session_name,
                'interaction_type': interaction_type,
                'run_number': run_number,
                'agent': agent,
                'behav_type': behav_type,
                'from': 'object',
                'to': 'elsewhere',
                'binary_timeline': merged_timeline.tolist()
            }])
        ], ignore_index=True)
        logger.debug(f"Merged 'from' saccade object timelines for group {group_idx}.")
    # Merge for the `to` field
    to_mask = group_df['to'].isin(['left_nonsocial_object', 'right_nonsocial_object'])
    to_df = group_df[to_mask]
    group_df = group_df[~to_mask]  # Remove left/right rows
    if not to_df.empty:
        merged_timeline = np.bitwise_or.reduce(np.stack(to_df['binary_timeline'].values))
        group_df = pd.concat([
            group_df,
            pd.DataFrame([{
                'session_name': session_name,
                'interaction_type': interaction_type,
                'run_number': run_number,
                'agent': agent,
                'behav_type': behav_type,
                'from': 'elsewhere',
                'to': 'object',
                'binary_timeline': merged_timeline.tolist()
            }])
        ], ignore_index=True)
        logger.debug(f"Merged 'to' saccade object timelines for group {group_idx}.")
    return group_df



def compute_behavioral_cross_correlations(
    binary_behav_timeseries_df, params, shuffled=False, serial=False, remake_results=True
):
    """
    Main function to compute behavioral cross-correlations, supporting both regular
    and shuffled modes. Defaults to regular computation unless 'shuffled' is True.
    """
    logger.info("Starting behavioral cross-correlation computation.")
    output_dir = os.path.join(
        params.get("processed_data_dir"), "crosscorr_outputs_shuffled" if shuffled else "crosscorr_outputs"
    )
    os.makedirs(output_dir, exist_ok=True)
    remake_results = False
    # Generate all cross-combinations for behavior types
    if remake_results:
        unique_from = binary_behav_timeseries_df['from'].unique()
        unique_to = binary_behav_timeseries_df['to'].unique()
        all_behavior_combinations = _generate_behavior_combinations(unique_from, unique_to)
        logger.debug(f"Generated {len(all_behavior_combinations)} behavior combinations.")
        # Group the data by session, interaction type, and run number
        grouped = binary_behav_timeseries_df.groupby(['session_name', 'interaction_type', 'run_number'])
        logger.info(f"Processing {len(grouped)} session groups.")
        num_cpus = params.get('num_cpus', 1)
        outer_pool_size = max(1, num_cpus // (3 if shuffled else 2))
        if shuffled:
            # Handle shuffled case with HPC job submission
            logger.info("Submitting jobs for shuffled cross-correlation computation.")
            binary_timeseries_file_path = os.path.join(params.get("processed_data_dir"), "binary_behav_timeseries.pkl")
            # Initialize HPCShuffledCrossCorr
            hpc_shuffled_cross_corr = HPCShuffledCrossCorr(params)
            # Generate the job file
            num_cpus = outer_pool_size
            shuffle_count = params.get("shuffle_count", 100)
            job_file_path = hpc_shuffled_cross_corr.generate_job_file(
                groups=grouped.groups.keys(),
                binary_timeseries_file_path=binary_timeseries_file_path,
                num_cpus=num_cpus,
                num_shuffles=shuffle_count,
                cross_combinations=all_behavior_combinations,
                output_dir=output_dir,
            )
            # Submit the job array
            hpc_shuffled_cross_corr.submit_job_array(job_file_path, num_cpus)
        else:
            # Handle regular or serial computation
            args = [
                (group_keys, group_df.to_dict(orient="list"), all_behavior_combinations, output_dir, None, shuffled)
                for group_keys, group_df in grouped
            ]
            if serial:
                logger.info("Running computations serially for debugging.")
                for arg_tuple in tqdm(args, desc="Calculating behavioral crosscorrelations for runs in serial"):
                    _compute_crosscorrelations_for_group(arg_tuple)
            else:
                num_cpus = params.get('num_cpus', 1)
                outer_pool_size = max(1, num_cpus // (3 if shuffled else 2))
                logger.info(f"Using {outer_pool_size} tasks with {num_cpus} CPUs allocated.")
                with Pool(outer_pool_size) as pool:
                    list(tqdm(pool.imap(_parallel_crosscorrelation_task, args), total=len(args), desc="Calculating behavioral crosscorrelations for runs in parallel"))
        logger.info(f"Behavioral cross-correlation results saved to {output_dir}.")
        return _collate_crosscorrelation_results(output_dir)
    else:
        return _collate_crosscorrelation_results(output_dir)


def _collate_crosscorrelation_results(output_dir):
    """
    Load cross-correlation results from the output directory and collate them into a single DataFrame.
    Args:
        output_dir (str): Path to the directory containing the cross-correlation result files.
    Returns:
        pd.DataFrame: DataFrame containing all collated results.
    """
    logger.info(f"Collating results from {output_dir}.")
    result_files = glob.glob(os.path.join(output_dir, "*.pkl"))
    logger.info(f"Found {len(result_files)} result files.")
    results = []
    for file_path in tqdm(result_files, desc="Loading result files"):
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    logger.info(f"Loaded {len(results)} cross-correlation results.")
    return pd.DataFrame(results)


def _generate_behavior_combinations(unique_from, unique_to):
    """
    Generate all possible behavior combinations for m1 and m2:
    - Fixations: 'from' == 'to', including an additional ('any', 'any') entry.
    - Saccades:
      - Older version preserved with 'any' and specific 'to' or 'from'.
      - Includes an additional ('any', 'any') entry for each monkey.
    Args:
        unique_from (array-like): Unique 'from' values.
        unique_to (array-like): Unique 'to' values.
    Returns:
        list: List of all possible behavior combinations for m1 and m2.
    """
    logger.info("Generating behavior combinations for cross-correlation.")
    # Remove 'all' from unique_from and unique_to
    unique_from = [loc for loc in unique_from if loc != "all"]
    unique_to = [loc for loc in unique_to if loc != "all"]
    m1_combinations = []
    m2_combinations = []
    # Fixation combinations (from == to)
    for loc in unique_from:
        m1_combinations.append(("fixation", loc, loc))
        m2_combinations.append(("fixation", loc, loc))
    # Add ('any', 'any') for fixation
    m1_combinations.append(("fixation", "any", "any"))
    m2_combinations.append(("fixation", "any", "any"))
    # Saccade combinations (older logic preserved)
    for to in unique_to:
        m1_combinations.append(("saccade", "any", to))
        m2_combinations.append(("saccade", "any", to))
    for from_ in unique_from:
        m1_combinations.append(("saccade", from_, "any"))
        m2_combinations.append(("saccade", from_, "any"))
    # Add ('any', 'any') for saccades
    m1_combinations.append(("saccade", "any", "any"))
    m2_combinations.append(("saccade", "any", "any"))
    # Cartesian product for m1 and m2 combinations
    cross_combinations = [
        (m1_type, m1_from, m1_to, m2_type, m2_from, m2_to)
        for m1_type, m1_from, m1_to in m1_combinations
        for m2_type, m2_from, m2_to in m2_combinations
    ]
    logger.info(f"Generated {len(cross_combinations)} behavior combinations.")
    return cross_combinations


def _parallel_crosscorrelation_task(arg_tuple):
    """
    Wrapper function for parallel cross-correlation task.
    Args:
        arg (tuple): Arguments unpacked for the `_compute_crosscorrelations_for_group` function.
    Returns:
        None
    """
    # Pass the entire tuple to the function
    _compute_crosscorrelations_for_group(arg_tuple)


def _compute_crosscorrelations_for_group(arg_tuple):
    """
    Compute cross-correlations for a single session group and save the results.
    Args:
        args (tuple): Contains group data and parameters for computation.
        shuffled (bool): Whether to compute shuffled cross-correlation distributions.
    Returns:
        None: Results are saved to disk.
    """
    group_keys, group_dict, cross_combinations, output_dir, shuffle_count, shuffled = arg_tuple
    session_name, interaction_type, run_number = group_keys
    group_df = pd.DataFrame(group_dict)
    # Compute cross-correlations for all combinations
    results = []
    for comb in cross_combinations:
        result = __compute_single_crosscorrelation_combination(
            (comb, group_df, shuffle_count, session_name, interaction_type, run_number),
            shuffled
        )
        if result is not None:
            results.append(result)
    # Save results to file
    sanitized_session_name = session_name.replace(" ", "_")
    sanitized_interaction_type = interaction_type.replace(" ", "_")
    filename = f"{sanitized_session_name}__{sanitized_interaction_type}__run_{run_number}"
    filename += "_shuffled.pkl" if shuffled else ".pkl"
    output_file = os.path.join(output_dir, filename)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    logger.debug(f"Saved results for group {group_keys} to {output_file}.")


def __compute_single_crosscorrelation_combination(args, shuffled=False):
    """
    Compute cross-correlation (regular or shuffled) for a single behavior combination.
    Args:
        args (tuple): Contains combination details, m1 and m2 data, shuffle count,
                      and group identifiers.
        shuffled (bool): Whether to compute shuffled cross-correlation distributions.
    Returns:
        dict or None: Result dictionary with cross-correlation data, or None if data is missing.
    """
    comb, group_df, shuffle_count, session_name, interaction_type, run_number = args
    behav_type_m1, m1_from, m1_to, behav_type_m2, m2_from, m2_to = comb
    # Extract binary timelines for both agents
    binary_timeline_m1 = ___extract_binary_vector(behav_type_m1, m1_from, m1_to, group_df, "m1")
    binary_timeline_m2 = ___extract_binary_vector(behav_type_m2, m2_from, m2_to, group_df, "m2")
    if binary_timeline_m1 is None or binary_timeline_m2 is None:
        return None
    if shuffled:
        # Shuffled cross-correlation computation
        return ___compute_shuffled_crosscorrelation(
            binary_timeline_m1, binary_timeline_m2, shuffle_count, session_name,
            interaction_type, run_number, behav_type_m1, m1_from, m1_to, behav_type_m2, m2_from, m2_to
        )
    else:
        # Regular cross-correlation computation
        return ___compute_regular_crosscorrelation(
            binary_timeline_m1, binary_timeline_m2, session_name,
            interaction_type, run_number, behav_type_m1, m1_from, m1_to, behav_type_m2, m2_from, m2_to
        )


def ___extract_binary_vector(behav_type, from_, to_, data, agent):
    """
    Extract binary vectors based on behavior type, agent, and 'from'-'to' combinations.
    Args:
        behav_type (str): Behavior type ('fixation' or 'saccade').
        from_ (str): 'From' field value.
        to_ (str): 'To' field value.
        data (pd.DataFrame): DataFrame containing behavioral data.
        agent (str): The agent ('m1' or 'm2') whose data to extract.
    Returns:
        np.array or None: Binary vector or None if no matching data.
    """
    # Build the filter condition based on behavior type
    if from_ == "any" and to_ == "any":
        # No need for 'from' or 'to' filtering; take all cases for the agent
        filter_cond = (data['behav_type'] == behav_type) & (data['agent'] == agent)
    elif behav_type == "fixation":
        filter_cond = (
            (data['behav_type'] == behav_type) &
            (data['from'] == from_) &
            (data['to'] == to_) &
            (data['agent'] == agent)
        )
    elif behav_type == "saccade":
        if from_ == "any":
            filter_cond = (
                (data['behav_type'] == behav_type) &
                (data['to'] == to_) &
                (data['agent'] == agent)
            )
        elif to_ == "any":
            filter_cond = (
                (data['behav_type'] == behav_type) &
                (data['from'] == from_) &
                (data['agent'] == agent)
            )
        else:
            filter_cond = (
                (data['behav_type'] == behav_type) &
                (data['from'] == from_) &
                (data['to'] == to_) &
                (data['agent'] == agent)
            )
    else:
        logger.warning(f"Unsupported behavior type: {behav_type}")
        return None
    # Apply the filter and extract binary timelines
    filtered_timelines = data[filter_cond]['binary_timeline'].apply(np.array)
    valid_timelines = [arr for arr in filtered_timelines if isinstance(arr, np.ndarray) and arr.ndim == 1]
    # Combine timelines with a bitwise OR if valid timelines exist
    if valid_timelines:
        try:
            combined_timelines = np.bitwise_or.reduce(np.vstack(valid_timelines))
            return combined_timelines
        except ValueError as e:
            logger.error(f"Error combining timelines: {e}")
            logger.debug(f"Valid timelines: {valid_timelines}")
    else:
        logger.debug(f"No valid binary timelines found after filtering for: {agent}: {behav_type} from {from_} to {to_}")
    return None


def ___compute_regular_crosscorrelation(
    binary_timeline_m1, binary_timeline_m2, session_name, interaction_type, run_number,
    m1_type, m1_from, m1_to, m2_type, m2_from, m2_to
):
    """
    Compute regular cross-correlation for a single combination.
    Args:
        binary_timeline_m1 (np.array): Binary timeline for m1.
        binary_timeline_m2 (np.array): Binary timeline for m2.
        session_name (str): Session name.
        interaction_type (str): Interaction type.
        run_number (int): Run number.
        m1_type, m1_from, m1_to (str): Behavior type, 'from', and 'to' for m1.
        m2_type, m2_from, m2_to (str): Behavior type, 'from', and 'to' for m2.
    Returns:
        dict: Cross-correlation result.
    """
    crosscorr = fftconvolve(binary_timeline_m1, binary_timeline_m2[::-1], mode='full')
    lags = np.arange(-len(binary_timeline_m1) + 1, len(binary_timeline_m2))
    return {
        'session_name': session_name,
        'interaction_type': interaction_type,
        'run_number': run_number,
        'm1_behav_type': m1_type,
        'm1_from': m1_from,
        'm1_to': m1_to,
        'm2_behav_type': m2_type,
        'm2_from': m2_from,
        'm2_to': m2_to,
        'lag': lags[np.argmax(crosscorr)],
        'max_correlation': crosscorr.max(),
        'crosscorr': crosscorr.tolist()
    }


def ___compute_shuffled_crosscorrelation(
    binary_timeline_m1, binary_timeline_m2, shuffle_count, session_name,
    interaction_type, run_number, m1_type, m1_from, m1_to, m2_type, m2_from, m2_to
):
    """
    Compute shuffled cross-correlations for a single combination.
    Args:
        binary_timeline_m1 (np.array): Binary timeline for m1.
        binary_timeline_m2 (np.array): Binary timeline for m2.
        shuffle_count (int): Number of shuffles for the cross-correlation.
        session_name (str): Session name.
        interaction_type (str): Interaction type.
        run_number (int): Run number.
        m1_type, m1_from, m1_to (str): Behavior type, 'from', and 'to' for m1.
        m2_type, m2_from, m2_to (str): Behavior type, 'from', and 'to' for m2.
    Returns:
        dict: Dictionary containing shuffled cross-correlation results.
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        crosscorr_shuffles = list(
            tqdm(
                executor.map(
                    lambda _: ____shuffle_and_compute(binary_timeline_m1, binary_timeline_m2),
                    range(shuffle_count)
                ),
                total=shuffle_count,
                desc="Shuffling and computing cross-correlations"
            )
        )
    crosscorr_shuffles = np.array(crosscorr_shuffles)
    mean_crosscorr = np.mean(crosscorr_shuffles, axis=0)
    std_crosscorr = np.std(crosscorr_shuffles, axis=0)
    lags = np.arange(-len(binary_timeline_m1) + 1, len(binary_timeline_m2))
    return {
        'session_name': session_name,
        'interaction_type': interaction_type,
        'run_number': run_number,
        'm1_behav_type': m1_type,
        'm1_from': m1_from,
        'm1_to': m1_to,
        'm2_behav_type': m2_type,
        'm2_from': m2_from,
        'm2_to': m2_to,
        'lags': lags.tolist(),
        'mean_crosscorr': mean_crosscorr.tolist(),
        'std_crosscorr': std_crosscorr.tolist()
    }


def ____shuffle_and_compute(binary_timeline_m1, binary_timeline_m2):
    """
    Perform a single shuffle and compute cross-correlation.
    Args:
        binary_timeline_m1 (np.array): Binary timeline for m1.
        binary_timeline_m2 (np.array): Binary timeline for m2.
    Returns:
        np.array: Cross-correlation result for the shuffle.
    """
    np.random.shuffle(binary_timeline_m1)
    np.random.shuffle(binary_timeline_m2)
    return fftconvolve(binary_timeline_m1, binary_timeline_m2[::-1], mode='full')

























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





























