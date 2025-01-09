import os
import random
import logging
import pandas as pd
import pickle
import numpy as np
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from multiprocessing import Pool, cpu_count

import curate_data
import load_data
import util
import hpc_fix_and_saccade_detector
import fixation_detector
import saccade_detector

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


def main():
    logger.info("Starting the main function")
    # Initialize params with data paths
    params = _initialize_params()
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
        eye_mvm_behav_df.to_pickle(eye_mvm_behav_df_file_path)
        logger.info("Fix and saccade saved to combined df in %s", eye_mvm_behav_df_file_path)
    else:
        logger.info("Loading eye_mvm_behav_df")
        eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)

    logger.info(f"Annotating fix and saccade locations in eye mvm behav df.")
    eye_mvm_behav_df = _update_fixation_and_saccade_locations_in_eye_mvm_dataframe(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df)
    eye_mvm_behav_df.to_pickle(eye_mvm_behav_df_file_path)
    logger.info(f"Fix and saccade with positions annotated saved to combined df in {eye_mvm_behav_df_file_path}")
    pdb.set_trace()
    return 0


def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'recompute_sparse_nan_removed_gaze_data': False,
        'remake_eye_mvm_df_from_gaze_data': False,
        'try_using_single_run': False,
        'recompute_fix_and_saccades_through_hpc_jobs': False,
        'recompute_fix_and_saccades': True,
        'is_grace': False,
        'hpc_job_output_subfolder': 'single_run_fix_sacc_detection_results'
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    params = curate_data.add_raw_data_dir_to_params(params)
    params = curate_data.add_paths_to_all_data_files_to_params(params)
    params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(params)
    logger.info("Parameters initialized successfully")
    return params



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
    if params.get('try_using_single_run'):
        random_run = random.choice(df_keys_for_tasks)
        random_session, random_interaction_type, random_run_num, random_agent, _ = random_run
        df_keys_for_tasks = [random_run]
        logger.warning(
            "Testing using positions data from a random single run: %s, %s, %s, %s",
            random_session, random_interaction_type, random_run_num, random_agent
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
            logger.info("Updating fix/saccade results for: %s, %s, %s, %s", session, interaction_type, run, agent)
            run_str = str(run)
            fix_path = os.path.join(params['processed_data_dir'], hpc_data_subfolder, f'fixation_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            sacc_path = os.path.join(params['processed_data_dir'], hpc_data_subfolder, f'saccade_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            fix_indices = None
            sacc_indices = None
            microsacc_inds = None
            if os.path.exists(fix_path):
                with open(fix_path, 'rb') as f:
                    fix_indices = pickle.load(f)
            if os.path.exists(sacc_path):
                with open(sacc_path, 'rb') as f:
                    sacc_indices, microsacc_inds = pickle.load(f)
            # Verification: Ensure ascending order
            assert np.all(np.diff(fix_indices[:, 0]) >= 0), "Fixation start-stops are not in ascending order."
            assert np.all(np.diff(sacc_indices[:, 0]) >= 0), "Saccade start-stops are not in ascending order."
            assert np.all(np.diff(microsacc_inds[:, 0]) >= 0), "Microsaccade start-stops are not in ascending order."
            eye_mvm_behav_rows.append({
                'session_name': session,
                'interaction_type': interaction_type,
                'run_number': run,
                'agent': agent,
                'fixation_start_stop': fix_indices,
                'saccade_start_stop': sacc_indices,
                'microsaccade_start_stop': microsacc_inds
            })
    else:
        for task in df_keys_for_tasks:
            session, interaction_type, run, agent, positions = task
            fixation_start_stop_inds, saccades_start_stop_inds, microsaccades_start_stop_inds = \
                _detect_fixations_saccades_and_microsaccades_in_run(positions, session)
            # Verification: Ensure ascending order
            assert np.all(np.diff(fix_indices[:, 0]) >= 0), "Fixation start-stops are not in ascending order."
            assert np.all(np.diff(sacc_indices[:, 0]) >= 0), "Saccade start-stops are not in ascending order."
            assert np.all(np.diff(microsacc_inds[:, 0]) >= 0), "Microsaccade start-stops are not in ascending order."
            eye_mvm_behav_rows.append({
                'session_name': session,
                'interaction_type': interaction_type,
                'run_number': run,
                'agent': agent,
                'fixation_start_stop': fixation_start_stop_inds,
                'saccade_start_stop': saccades_start_stop_inds,
                'microsaccade_start_stop': microsaccades_start_stop_inds
            })
    logger.info("Fixation and saccade detection completed")
    return pd.DataFrame(eye_mvm_behav_rows)


def _detect_fixations_saccades_and_microsaccades_in_run(positions, session_name):
    logger.info("Detecting fixations, saccades, and microsaccades for session: %s", session_name)
    non_nan_chunks, chunk_start_indices = __extract_non_nan_chunks(positions)
    num_cpus = cpu_count()
    logger.info("Detected %d CPUs for parallel processing", num_cpus)
    args = [(chunk, start_ind, session_name) for chunk, start_ind in zip(non_nan_chunks, chunk_start_indices)]
    parallel_threads = min(8, num_cpus)
    with Pool(processes=parallel_threads) as pool:
        results = pool.map(__detect_fix_sacc_micro_in_chunk, args)
    all_fix_start_stops = np.empty((0, 2), dtype=int)
    all_sacc_start_stops = np.empty((0, 2), dtype=int)
    all_microsacc_start_stops = np.empty((0, 2), dtype=int)
    for fix_stops, sacc_stops, micro_stops in results:
        all_fix_start_stops = np.concatenate((all_fix_start_stops, fix_stops), axis=0)
        all_sacc_start_stops = np.concatenate((all_sacc_start_stops, sacc_stops), axis=0)
        all_microsacc_start_stops = np.concatenate((all_microsacc_start_stops, micro_stops), axis=0)
    # Verification: Ensure ascending order
    assert np.all(np.diff(all_fix_start_stops[:, 0]) >= 0), "Fixation start-stops are not in ascending order."
    assert np.all(np.diff(all_sacc_start_stops[:, 0]) >= 0), "Saccade start-stops are not in ascending order."
    assert np.all(np.diff(all_microsacc_start_stops[:, 0]) >= 0), "Microsaccade start-stops are not in ascending order."
    logger.info("Fixations, saccades, and microsaccades detection completed for session: %s", session_name)
    return all_fix_start_stops, all_sacc_start_stops, all_microsacc_start_stops


def __detect_fix_sacc_micro_in_chunk(args):
    position_chunk, start_ind, session_name = args
    logger.debug("Detecting fix/sacc/microsacc in chunk starting at index %d for session: %s", start_ind, session_name)
    fixation_start_stop_indices = fixation_detector.detect_fixation_in_position_array(position_chunk, session_name)
    fixation_start_stop_indices += start_ind
    saccades_start_stop_inds, microsaccades_start_stop_inds = \
        saccade_detector.detect_saccades_and_microsaccades_in_position_array(position_chunk, session_name)
    saccades_start_stop_inds += start_ind
    microsaccades_start_stop_inds += start_ind
    return fixation_start_stop_indices, saccades_start_stop_inds, microsaccades_start_stop_inds


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
        fixation_labels = []
        for fixation in fixations:
            start_idx, stop_idx = fixation
            mean_position = positions[start_idx:stop_idx + 1].mean(axis=0)
            fixation_labels.append(__determine_roi_of_location(mean_position, roi_rects))
        # Process saccades
        saccades = row['saccade_start_stop']
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



if __name__ == "__main__":
    main()
