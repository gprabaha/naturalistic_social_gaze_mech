import os
import random
import logging
import pandas as pd
import pickle
import numpy as np
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm
from datetime import datetime

import curate_data
import load_data
import util
import hpc_fix_and_saccade_detector
import cluster_fix

import pdb



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'recompute_sparse_nan_removed_gaze_data': False,
        'remake_eye_mvm_df_from_gaze_data': False,
        'try_using_single_run': False,
        'test_specific_runs': False,
        'recompute_fix_and_saccades_through_hpc_jobs': False,
        'recompute_fix_and_saccades': False,
        'plot_eye_mvm_behav': False,
        'plot_gaze_event_dur_dist': False,
        'is_grace': False,
        'hpc_job_output_subfolder': 'single_run_fix_sacc_detection_results'
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    logger.info("Parameters initialized successfully")
    return params



def main():
    logger.info("Starting the main function")
    # Initialize params with data paths
    params = _initialize_params()

    ## Get the gaze data, filter out object ROIs for M2 and filter out sparse NaNs
    sparse_nan_removed_sync_gaze_data_df_filepath = os.path.join(params['processed_data_dir'], 'sparse_nan_removed_sync_gaze_data_df.pkl')
    if params.get('recompute_sparse_nan_removed_gaze_data', False):
        # Load synchronized gaze data
        synchronized_gaze_data_file_path = os.path.join(params['processed_data_dir'], 'synchronized_gaze_data_df.pkl')
        synchronized_gaze_data_df = load_data.get_data_df(synchronized_gaze_data_file_path)
        # Remove object ROIs from M2 agent's data
        synchronized_gaze_data_df = _remove_object_rois_for_m2(synchronized_gaze_data_df)
        logger.info("Loaded synchronized gaze data from %s", synchronized_gaze_data_file_path)
        # Sparse NaN removal from each run's positions
        sparse_nan_removed_sync_gaze_df = synchronized_gaze_data_df.copy()
        logger.info("Interpolating sparse nans from whole gaze data")
        sparse_nan_removed_sync_gaze_df['positions'] = sparse_nan_removed_sync_gaze_df['positions'].apply(
            lambda pos: _interpolate_nans_in_positions_with_sliding_window(pos)
        )
        sparse_nan_removed_sync_gaze_df.to_pickle(sparse_nan_removed_sync_gaze_data_df_filepath)
        logger.info(f"Saved sparse_nan_removed_sync_gaze_df to {sparse_nan_removed_sync_gaze_data_df_filepath}")
    else:
        logger.info("Loading sparse_nan_removed_sync_gaze_df")
        sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)

    ## Detect fixations, saccades, and microsaccades from position data and then annotate ROIs of behavior
    eye_mvm_behav_df_file_path = os.path.join(params['processed_data_dir'], 'eye_mvm_behav_df.pkl')
    if params.get('remake_eye_mvm_df_from_gaze_data', False):
        logger.info("Remaking eye_mvm_df from gaze data.")
        # Separate sessions and runs to process separately
        df_keys_for_tasks = _prepare_tasks(sparse_nan_removed_sync_gaze_df, params)
        # Save params to a file
        _save_params(params)
        # Process fixation and saccade detection
        eye_mvm_behav_df = _process_fixations_and_saccades(df_keys_for_tasks, params)
        # Log results
        logger.info("Detection completed for both agents.")
        logger.info(f"Annotating fix and saccade locations in eye mvm behav df.")
        eye_mvm_behav_df = _update_fixation_and_saccade_locations_in_eye_mvm_dataframe(
            eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df)
        eye_mvm_behav_df = _align_and_correct_consecutive_gaze_event_labels(eye_mvm_behav_df)
        eye_mvm_behav_df = _add_run_length_column_to_behav_df(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df)
        eye_mvm_behav_df.to_pickle(eye_mvm_behav_df_file_path)
        logger.info(f"Fix and saccade with positions annotated saved to combined df in {eye_mvm_behav_df_file_path}")
    else:
        logger.info("Loading eye_mvm_behav_df")
        eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
        eye_mvm_behav_df = _align_and_correct_consecutive_gaze_event_labels(eye_mvm_behav_df)
        # eye_mvm_behav_df.to_pickle(eye_mvm_behav_df_file_path)
    if params.get('plot_eye_mvm_behav', True):
        ## Plot fixation and saccade behavior for each run
        _plot_eye_mvm_behav_for_each_run(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params)

    if params.get('plot_gaze_event_dur_dist', True):
        _plot_gaze_event_duration_distributions(eye_mvm_behav_df, params)


def _remove_object_rois_for_m2(dataframe):
    """
    Modify the 'roi_rects' column in a dataframe based on the 'agent' column.
    - Retains all ROI rectangles for rows where the 'agent' is 'm1'.
    - Removes 'object' ROIs from the 'roi_rects' dictionary for rows where the 'agent' is 'm2'.
    Parameters:
        dataframe (pd.DataFrame): A dataframe containing a 'roi_rects' column with ROI data
                                  and an 'agent' column indicating the agent ('m1' or 'm2').
    Returns:
        pd.DataFrame: The modified dataframe with updated 'roi_rects' values.
    """
    # Apply changes to the 'roi_rects' column based on the 'agent' value
    dataframe['roi_rects'] = dataframe.apply(
        lambda row: {
            roi_name: roi_coords 
            for roi_name, roi_coords in row['roi_rects'].items()
            if row['agent'] == 'm1' or 'object' not in roi_name  # Retain only relevant ROIs
        },
        axis=1
    )
    return dataframe


def _interpolate_nans_in_positions_with_sliding_window(positions, window_size=10, max_nans=3):
    logger.debug("Starting NaN interpolation with sliding window")
    positions = positions.copy()
    num_points, num_dims = positions.shape
    stride = max_nans
    global_nan_mask = np.isnan(positions).any(axis=1)
    for start in range(0, num_points - window_size + 1, stride):
        end = start + window_size
        window_nan_mask = global_nan_mask[start:end]
        nan_count = np.sum(window_nan_mask)
        if 0 < nan_count <= max_nans:
            window = positions[start:end].copy()
            for col in range(num_dims):
                col_values = window[:, col]
                valid_indices = np.where(~np.isnan(col_values))[0]
                valid_values = col_values[valid_indices]
                if len(valid_indices) > 1:
                    interp_func = interp1d(
                        valid_indices, valid_values, kind='cubic', bounds_error=False, fill_value="extrapolate"
                    )
                    nan_indices = np.where(window_nan_mask)[0]
                    interpolated_values = interp_func(nan_indices)
                    col_values[nan_indices] = interpolated_values
            positions[start:end] = window
    logger.debug("Completed NaN interpolation")
    return positions


def _prepare_tasks(synchronized_gaze_data_df, params):
    logger.info("Preparing tasks for fixation and saccade detection")
    df_keys_for_tasks = synchronized_gaze_data_df[
        ['session_name', 'interaction_type', 'run_number', 'agent', 'positions']
    ].values.tolist()
    if params.get('try_using_single_run', False):
        random_run = random.choice(df_keys_for_tasks)
        random_session, random_interaction_type, random_run_num, random_agent, _ = random_run
        df_keys_for_tasks = [random_run]
        logger.warning(
            "Testing using positions data from a random single run: %s, %s, %s, %s",
            random_session, random_interaction_type, random_run_num, random_agent
        )
    elif params.get('test_specific_runs', False):
        specific_runs = [
            ('01072019', 'interactive', 2, 'm1'),
            ('08292018', 'interactive', 8, 'm2')
        ]
        df_keys_for_tasks = [
            task for task in df_keys_for_tasks 
            if (task[0], task[1], task[2], task[3]) in specific_runs
        ]
        logger.warning(
            "Testing fixation detection for specific runs: %s",
            specific_runs
        )
    logger.info("Tasks prepared successfully")
    return df_keys_for_tasks


def _save_params(params):
    params_file_path = os.path.join(params['processed_data_dir'], 'params.pkl')
    with open(params_file_path, 'wb') as f:
        pickle.dump(params, f)
    logger.info("Pickle dumped params to %s", params_file_path)


def _process_fixations_and_saccades(df_keys_for_tasks, params):
    logger.info("Starting fixation and saccade detection")
    eye_mvm_behav_rows = []
    params_file_path = os.path.join(params['processed_data_dir'], 'params.pkl')
    if params.get('recompute_fix_and_saccades_through_hpc_jobs', False):
        if params.get('recompute_fix_and_saccades', False):
            detector = hpc_fix_and_saccade_detector.HPCFixAndSaccadeDetector(params)
            job_file_path = detector.generate_job_file(df_keys_for_tasks, params_file_path)
            detector.submit_job_array(job_file_path)
            logger.info("Submitted HPC job array for fixation and saccade detection")
        hpc_data_subfolder = params.get('hpc_job_output_subfolder', '')
        for task in df_keys_for_tasks:
            session, interaction_type, run, agent, _ = task
            logger.info("Appending fix/saccade results for: %s, %s, %s, %s", session, interaction_type, run, agent)
            run_str = str(run)
            fix_path = os.path.join(params['processed_data_dir'], hpc_data_subfolder, f'fixation_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            sacc_path = os.path.join(params['processed_data_dir'], hpc_data_subfolder, f'saccade_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            fix_indices = None
            sacc_indices = None
            if os.path.exists(fix_path):
                with open(fix_path, 'rb') as f:
                    fix_indices = pickle.load(f)
            if os.path.exists(sacc_path):
                with open(sacc_path, 'rb') as f:
                    sacc_indices = pickle.load(f)
            eye_mvm_behav_rows.append({
                'session_name': session,
                'interaction_type': interaction_type,
                'run_number': run,
                'agent': agent,
                'fixation_start_stop': fix_indices,
                'saccade_start_stop': sacc_indices
            })
    else:
        for task in df_keys_for_tasks:
            session, interaction_type, run, agent, positions = task
            fixation_start_stop_inds, saccades_start_stop_inds = \
                _detect_fixations_and_saccades_in_run(positions, session, True)
            eye_mvm_behav_rows.append({
                'session_name': session,
                'interaction_type': interaction_type,
                'run_number': run,
                'agent': agent,
                'fixation_start_stop': fixation_start_stop_inds,
                'saccade_start_stop': saccades_start_stop_inds
            })
    logger.info("Fixation and saccade detection completed")
    return pd.DataFrame(eye_mvm_behav_rows)


def _detect_fixations_and_saccades_in_run(positions, session_name, use_parallel=True):
    logger.info("Detecting fixations, saccades, and microsaccades for session: %s", session_name)
    non_nan_chunks, chunk_start_indices = __extract_non_nan_chunks(positions)
    args = [(chunk, start_ind) for chunk, start_ind in zip(non_nan_chunks, chunk_start_indices)]
    if use_parallel:
        num_cpus = cpu_count()
        parallel_threads = min(16, num_cpus)
        logger.info("Using %d CPUs for parallel processing", parallel_threads)
        with Pool(processes=parallel_threads) as pool:
            results = pool.map(__detect_fix_sacc_in_chunk, args)
    else:
        logger.info("Running in serial mode")
        results = [__detect_fix_sacc_in_chunk(arg) for arg in args]
    all_fix_start_stops = np.empty((0, 2), dtype=int)
    all_sacc_start_stops = np.empty((0, 2), dtype=int)
    for fix_stops, sacc_stops in results:
        all_fix_start_stops = np.concatenate((all_fix_start_stops, fix_stops), axis=0)
        all_sacc_start_stops = np.concatenate((all_sacc_start_stops, sacc_stops), axis=0)
    # Combine both fixation and saccade intervals
    all_events = np.vstack((all_fix_start_stops, all_sacc_start_stops))
    # Sort events by start time
    all_events = all_events[np.argsort(all_events[:, 0])]
    # Check for overlaps
    for i in range(len(all_events) - 1):
        start1, end1 = all_events[i]
        start2, end2 = all_events[i + 1]
        assert end1 < start2, f"Overlap detected between ({start1}, {end1}) and ({start2}, {end2})"
    logger.info("Fixations and saccades detection completed for session: %s", session_name)
    return all_fix_start_stops, all_sacc_start_stops


def __extract_non_nan_chunks(positions):
    logger.debug("Extracting non-NaN chunks from positions")
    non_nan_chunks = []
    start_indices = []
    n = positions.shape[0]
    valid_mask = ~np.isnan(positions).any(axis=1)
    diff = np.diff(valid_mask.astype(int))
    chunk_starts = np.where(diff == 1)[0] + 1
    chunk_ends = np.where(diff == -1)[0] + 1
    if valid_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if valid_mask[-1]:
        chunk_ends = np.append(chunk_ends, n)
    for start, end in zip(chunk_starts, chunk_ends):
        non_nan_chunks.append(positions[start:end])
        start_indices.append(start)
    logger.debug("Extracted %d non-NaN chunks", len(non_nan_chunks))
    return non_nan_chunks, start_indices


def __detect_fix_sacc_in_chunk(args):
    position_chunk, start_ind = args
    logger.debug("Detecting fix/sacc/microsacc in chunk starting at index %d", start_ind)
    fixation_start_stop_indices, saccades_start_stop_inds = cluster_fix.detect_fixations_saccades(position_chunk)
    fixation_start_stop_indices += start_ind
    saccades_start_stop_inds += start_ind
    return fixation_start_stop_indices, saccades_start_stop_inds



def _update_fixation_and_saccade_locations_in_eye_mvm_dataframe(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df):
    """
    Annotate fixations and saccades with ROI labels for all session-interaction-run-agent combinations.
    Parameters:
    - eye_mvm_behav_df (DataFrame): DataFrame containing fixation and saccade information.
    - sparse_nan_removed_sync_gaze_df (DataFrame): DataFrame containing positions and ROI rects.
    Returns:
    - DataFrame: Updated eye_mvm_behav_df with additional columns `fixation_location`, `saccade_from`, and `saccade_to`.
    """
    # Create new columns to store results
    eye_mvm_behav_df['fixation_location'] = None
    eye_mvm_behav_df['saccade_from'] = None
    eye_mvm_behav_df['saccade_to'] = None
    # Group by session-interaction-run-agent combinations
    grouped = eye_mvm_behav_df.groupby(['session_name', 'interaction_type', 'run_number', 'agent'])
    for (session_name, interaction_type, run_number, agent), behav_group in grouped:
        # Ensure each behav_group has only one row
        if len(behav_group) != 1:
            raise ValueError(f"Expected only one row per group, but found {len(behav_group)} for group: {(session_name, interaction_type, run_number, agent)}")
        # Filter the gaze dataframe for the current group
        gaze_df = sparse_nan_removed_sync_gaze_df[
            (sparse_nan_removed_sync_gaze_df['session_name'] == session_name) &
            (sparse_nan_removed_sync_gaze_df['interaction_type'] == interaction_type) &
            (sparse_nan_removed_sync_gaze_df['run_number'] == run_number) &
            (sparse_nan_removed_sync_gaze_df['agent'] == agent)
        ]
        if gaze_df.empty:
            continue
        # Extract positions and ROI rects
        positions = gaze_df.iloc[0]['positions']
        roi_rects = gaze_df.iloc[0]['roi_rects']
        # Process the single row in behav_group
        row = behav_group.iloc[0]
        # Process fixations
        fixations = row['fixation_start_stop']
        if fixations is None or len(fixations) == 0:
            print(f"Empty or None fixation array found for group: {(session_name, interaction_type, run_number, agent)}")
            continue
        fixation_labels = []
        for fixation in fixations:
            start_idx, stop_idx = fixation
            mean_position = positions[start_idx:stop_idx + 1].mean(axis=0)
            fixation_labels.append(__determine_roi_of_location(mean_position, roi_rects))
        # Process saccades
        saccades = row['saccade_start_stop']
        if saccades is None or len(saccades) == 0:
            print(f"Empty or None saccade array found for group: {(session_name, interaction_type, run_number, agent)}")
            continue
        saccade_from = []
        saccade_to = []
        for saccade in saccades:
            start_idx, stop_idx = saccade
            start_position = positions[start_idx]
            end_position = positions[stop_idx]
            saccade_from.append(__determine_roi_of_location(start_position, roi_rects))
            saccade_to.append(__determine_roi_of_location(end_position, roi_rects))
        # Update the DataFrame with new columns
        eye_mvm_behav_df.at[behav_group.index[0], 'fixation_location'] = fixation_labels
        eye_mvm_behav_df.at[behav_group.index[0], 'saccade_from'] = saccade_from
        eye_mvm_behav_df.at[behav_group.index[0], 'saccade_to'] = saccade_to
    return eye_mvm_behav_df

def __determine_roi_of_location(position, roi_rects):
    """Determine if a position is within any ROI."""
    matching_rois = []
    for roi_name, rect in roi_rects.items():
        x_min, y_min, x_max, y_max = rect
        if x_min <= position[0] <= x_max and y_min <= position[1] <= y_max:
            matching_rois.append(roi_name)
    return matching_rois if matching_rois else ['out_of_roi']


def _align_and_correct_consecutive_gaze_event_labels(eye_mvm_behav_df):
    """
    Aligns fixation and saccade events in a gaze behavior dataframe by start times,
    then checks and corrects mismatches in location labels.
    Specifically:
    - Ensures fixations and saccades are ordered by start time.
    - If a fixation is followed by a saccade within 100 ms, checks if the fixation 
      location matches the saccade's starting location (`saccade_from`).
      - If one of them is `"out_of_roi"`, it is replaced with the other label.
    - If a saccade is followed by a fixation within 100 ms, checks if the saccade's 
      ending location (`saccade_to`) matches the fixation location.
      - Again, replaces `"out_of_roi"` if necessary.
    Parameters:
    ----------
    eye_mvm_behav_df : pandas.DataFrame
        Dataframe containing gaze behavior data, including fixation and saccade events.
    Modifies:
    ----------
    - Updates `"fixation_location"`, `"saccade_from"`, and `"saccade_to"` in-place.
    """
    for idx, row in tqdm(eye_mvm_behav_df.iterrows(), desc="Processing df row"):
        # Extract relevant columns
        session_name = row['session_name']
        agent = row['agent']
        fixations = row["fixation_start_stop"]
        saccades = row["saccade_start_stop"]
        fixation_locs = row["fixation_location"].copy()
        saccade_froms = row["saccade_from"].copy()
        saccade_tos = row["saccade_to"].copy()
        # Combine events into a single list for sorting
        events = [(start, end, "fixation", i) for i, (start, end) in enumerate(fixations)]
        events += [(start, end, "saccade", i) for i, (start, end) in enumerate(saccades)]
        events.sort(key=lambda x: x[0])  # Sort events by start time
        # Iterate through consecutive events
        for i in range(len(events) - 1):
            start1, end1, type1, index1 = events[i]
            start2, end2, type2, index2 = events[i + 1]
            # Case 1: Fixation → Saccade
            if type1 == "fixation" and type2 == "saccade":
                if start2 - end1 <= 100:  # Ensure the gap is within 100 ms
                    fixation_label = fixation_locs[index1]
                    saccade_from_label = saccade_froms[index2]
                    # Fix mismatch if one is "out_of_roi"
                    if set(fixation_label) != set(saccade_from_label):
                        if "out_of_roi" in fixation_label:
                            logger.info(
                                f"Updating fixation_location at index {index1} (session {session_name}, agent {agent}): "
                                f"{fixation_label} → {saccade_from_label}"
                            )
                            fixation_locs[index1] = saccade_from_label
                        if "out_of_roi" in saccade_from_label:
                            logger.info(
                                f"Updating saccade_from at index {index2} (session {session_name}, agent {agent}): "
                                f"{saccade_from_label} → {fixation_label}"
                            )
                            saccade_froms[index2] = fixation_label
            # Case 2: Saccade → Fixation
            elif type1 == "saccade" and type2 == "fixation":
                if start2 - end1 <= 100:  # Ensure the gap is within 100 ms
                    saccade_to_label = saccade_tos[index1]
                    fixation_label = fixation_locs[index2]
                    # Fix mismatch if one is "out_of_roi"
                    if set(saccade_to_label) != set(fixation_label):
                        if "out_of_roi" in fixation_label:
                            logger.info(
                                f"Updating fixation_location at index {index2} (session {session_name}, agent {agent}): "
                                f"{fixation_label} → {saccade_to_label}"
                            )
                            fixation_locs[index2] = saccade_to_label
                        if "out_of_roi" in saccade_to_label:
                            logger.info(
                                f"Updating saccade_to at index {index1} (session {session_name}, agent {agent}): "
                                f"{saccade_to_label} → {fixation_label}"
                            )
                            saccade_tos[index1] = fixation_label
        # Save the updated labels back into the DataFrame
        eye_mvm_behav_df.at[idx, "fixation_location"] = fixation_locs
        eye_mvm_behav_df.at[idx, "saccade_from"] = saccade_froms
        eye_mvm_behav_df.at[idx, "saccade_to"] = saccade_tos
    return eye_mvm_behav_df


def _add_run_length_column_to_behav_df(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df):
    # Create a lookup dictionary with keys as (session_name, run_number, agent) and values as run length
    run_length_dict = {
        (row.session_name, row.run_number, row.agent): len(row.positions)
        for _, row in sparse_nan_removed_sync_gaze_df.iterrows()
    }
    # Map the run length to the behavioral dataframe using the lookup dictionary
    eye_mvm_behav_df['run_length'] = eye_mvm_behav_df.apply(
        lambda row: run_length_dict.get((row.session_name, row.run_number, row.agent), None), axis=1
    )
    return eye_mvm_behav_df


def _plot_eye_mvm_behav_for_each_run(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params):
    # Get today's date for folder naming
    today_date = datetime.now().strftime("%Y-%m-%d")
    root_dir = os.path.join(
        params['root_data_dir'],
        "plots/eye_mvm_behavior",
        f"{today_date}"
    )
    os.makedirs(root_dir, exist_ok=True)
    session_groups = eye_mvm_behav_df.groupby(['session_name', 'interaction_type'])
    logger.info("Starting to generate behavioral plots...")
    for (session, interaction_type), group in tqdm(session_groups, desc="Plotting fix and sacc"):
        runs = group['run_number'].unique()
        fig, axs = plt.subplots(len(runs), 2, figsize=(12, 6 * len(runs)))
        if len(runs) == 1:
            axs = np.expand_dims(axs, axis=0)
        for row_idx, run in enumerate(runs):
            agents_data = {}
            for col_idx, agent in enumerate(['m1', 'm2']):
                ax = axs[row_idx, col_idx]
                ax.set_title(f"Run {run} - Agent {agent}")
                agent_data = group[(group['run_number'] == run) & (group['agent'] == agent)]
                gaze_data = sparse_nan_removed_sync_gaze_df[
                    (sparse_nan_removed_sync_gaze_df['session_name'] == session) &
                    (sparse_nan_removed_sync_gaze_df['interaction_type'] == interaction_type) &
                    (sparse_nan_removed_sync_gaze_df['run_number'] == run) &
                    (sparse_nan_removed_sync_gaze_df['agent'] == agent)
                ]
                if not gaze_data.empty:
                    agents_data[agent] = {
                        'fixation_start_stop': agent_data['fixation_start_stop'].iloc[0],
                        'saccade_start_stop': agent_data['saccade_start_stop'].iloc[0],
                        'positions': gaze_data['positions'].iloc[0],
                        'roi_rects': gaze_data['roi_rects'].iloc[0]
                    }
                roi_color = 'red'
                if agent in agents_data:
                    data = agents_data[agent]
                    fixation_start_stop = data['fixation_start_stop']
                    saccade_start_stop = data['saccade_start_stop']
                    positions = data['positions']
                    roi_rects = data['roi_rects']
                    total_time = len(positions)
                    time_norm = Normalize(vmin=0, vmax=total_time)
                    cmap = cm.viridis
                    all_events = []
                    all_events.extend([(start, stop, 'fixation') for start, stop in fixation_start_stop])
                    all_events.extend([(start, stop, 'saccade') for start, stop in saccade_start_stop])
                    all_events.sort(key=lambda x: x[0])
                    for start, stop, event_type in all_events:
                        if event_type == 'fixation':
                            fixation_positions = positions[start:stop]
                            mean_pos = fixation_positions.mean(axis=0)
                            color = cmap(time_norm((start + stop) // 2))
                            ax.plot(mean_pos[0], mean_pos[1], marker='o', color=color, zorder=2)
                        elif event_type == 'saccade':
                            saccade_positions = positions[start:stop]
                            color = cmap(time_norm((start + stop) // 2))
                            ax.plot(saccade_positions[:, 0], saccade_positions[:, 1], color=color, linewidth=2, zorder=2)
                    for roi_idx, (roi, rect) in enumerate(roi_rects.items()):
                        x, y, x_max, y_max = rect
                        ax.add_patch(
                            Rectangle((x, y), x_max - x, y_max - y, fill=False, edgecolor=roi_color, linewidth=1, zorder=3)
                        )
                    ax.invert_yaxis()
        plt.suptitle(f"Session: {session}, Interaction: {interaction_type}")
        plot_path = os.path.join(root_dir, f"{session}_{interaction_type}_fixation_saccade_timeseries.png")
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Saved plot: {plot_path}")
    logger.info("Behavioral plot generation completed.")



def _plot_gaze_event_duration_distributions(eye_mvm_behav_df, params):
    """
    Generates and saves histograms of fixation and saccade durations for each session.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    plot_dir = os.path.join(params['root_data_dir'], "plots", "gaze_event_durations", today_date)
    os.makedirs(plot_dir, exist_ok=True)
    session_groups = eye_mvm_behav_df.groupby("session_name")
    for session_name, session_df in tqdm(session_groups, desc="Plotting gaze event duration dist."):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        for agent_idx, agent in enumerate(["m1", "m2"]):
            agent_df = session_df[session_df["agent"] == agent]
            if agent_df.empty:
                logger.warning(f"Skipping {agent} for session {session_name}, no data found.")
                continue
            fixation_durations = []
            saccade_durations = []
            for _, row in agent_df.iterrows():
                fixations = row["fixation_start_stop"]
                saccades = row["saccade_start_stop"]
                fixation_durations.extend([end - start for start, end in fixations])
                saccade_durations.extend([end - start for start, end in saccades])
            # Plot histograms with max duration capped at 1500 ms
            axes[agent_idx].hist(fixation_durations, bins=50, alpha=0.75, label="Fixation")
            axes[agent_idx].hist(saccade_durations, bins=50, alpha=0.75, label="Saccade")
            axes[agent_idx].set_xlim(0, 1500)
            axes[agent_idx].set_title(f"{agent} Event Durations")
            axes[agent_idx].set_xlabel("Duration (ms)")
            axes[agent_idx].legend()
        axes[0].set_ylabel("Frequency")
        plt.suptitle(f"Gaze Event Durations - Session {session_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Save figure
        save_path = os.path.join(plot_dir, f"{session_name}_gaze_durations.png")
        plt.savefig(save_path)
        plt.close(fig)
    logger.info("Gaze event duration distribution plot generation completed.")




if __name__ == "__main__":
    main()
