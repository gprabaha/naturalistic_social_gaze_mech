
import os
import random
import pandas as pd
import pickle
import numpy as np
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view

import curate_data
import load_data
import util
import hpc_fix_and_saccade_detector
import fixation_detector
# import saccade_detector

import pdb

def main():
    # Initialize params with data paths
    params = _initialize_params()
    # Load synchronized gaze data
    synchronized_gaze_data_df = _load_synchronized_gaze_data(params)
    # Sparse NaN removal from each run's positions
    sparse_nan_removed_sync_gaze_df = synchronized_gaze_data_df.copy()
    sparse_nan_removed_sync_gaze_df['positions'] = sparse_nan_removed_sync_gaze_df['positions'].apply(
        lambda pos: _interpolate_nans_in_positions_with_sliding_window(pos)
    )
    # Separate sessions and runs to process separately
    df_keys_for_tasks = _prepare_tasks(synchronized_gaze_data_df, params)
    # Save params to a file
    _save_params(params)
    # Process fixation and saccade detection
    fixation_df, saccade_df = _process_fixations_and_saccades(df_keys_for_tasks, params)
    # Print results
    print("Detection completed for both agents.")


def _initialize_params():
    params = {
        'try_using_single_run': True,
        'recompute_fix_and_saccades_through_hpc_jobs': False
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    params = curate_data.add_raw_data_dir_to_params(params)
    params = curate_data.add_paths_to_all_data_files_to_params(params)
    params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(params)
    return params


def _load_synchronized_gaze_data(params):
    synchronized_gaze_data_file_path = os.path.join(params['processed_data_dir'], 'synchronized_gaze_data_df.pkl')
    return load_data.get_synchronized_gaze_data_df(synchronized_gaze_data_file_path)



def _interpolate_nans_in_positions_with_sliding_window(positions, window_size=10, max_nans=3):
    positions = positions.copy()
    num_points, num_dims = positions.shape
    stride = max_nans
    # Calculate the nan mask for the entire array
    global_nan_mask = np.isnan(positions).any(axis=1)
    for start in range(0, num_points - window_size + 1, stride):
        end = start + window_size
        # Use the global_nan_mask to count NaNs in the current window
        window_nan_mask = global_nan_mask[start:end]
        nan_count = np.sum(window_nan_mask)
        if 0 < nan_count <= max_nans:
            window = positions[start:end].copy()  # Extract and copy the current window
            for col in range(num_dims):
                col_values = window[:, col]
                valid_indices = np.where(~np.isnan(col_values))[0]
                valid_values = col_values[valid_indices]
                if len(valid_indices) > 1:  # Ensure there are enough points for interpolation
                    interp_func = interp1d(
                        valid_indices, valid_values, kind='cubic', bounds_error=False, fill_value="extrapolate"
                    )
                    nan_indices = np.where(window_nan_mask)[0]
                    interpolated_values = interp_func(nan_indices)
                    col_values[nan_indices] = interpolated_values
            positions[start:end] = window  # Update the positions array for the interpolated window
    return positions



def _prepare_tasks(synchronized_gaze_data_df, params):
    df_keys_for_tasks = synchronized_gaze_data_df[
        ['session_name', 'interaction_type', 'run_number', 'agent', 'positions']
    ].values.tolist()
    if params.get('try_using_single_run'):
        random_run = random.choice(df_keys_for_tasks)
        random_session, random_interaction_type, random_run_num, random_agent, _ = random_run
        df_keys_for_tasks = [random_run]
        print(f"!! Testing using positions data from a random single run: {random_session}, {random_interaction_type}, {random_run_num}, {random_agent} !!")
    return df_keys_for_tasks


def _save_params(params):
    params_file_path = os.path.join(params['processed_data_dir'], 'params.pkl')
    with open(params_file_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Pickle dumped params to {params_file_path}")


def _process_fixations_and_saccades(df_keys_for_tasks, params):
    fixation_rows = []
    saccade_rows = []
    if params.get('recompute_fix_and_saccades_through_hpc_jobs', False):
        if params.get('recompute_fix_and_saccades', False):
            detector = hpc_fix_and_saccade_detector.HPCFixAndSaccadeDetector(params)
            job_file_path = detector.generate_job_file(df_keys_for_tasks, params)
            detector.submit_job_array(job_file_path)
        hpc_data_subfolder = params.get('hpc_job_output_subfolder', '')
        for task in df_keys_for_tasks:
            session, interaction_type, run, agent, _ = task
            print(f'Updating fix/saccade results for: {session}, {interaction_type}, {run}, {agent}')
            run_str = str(run)
            fix_path = os.path.join(params['processed_data_dir'], hpc_data_subfolder, f'fixation_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            sacc_path = os.path.join(params['processed_data_dir'], hpc_data_subfolder, f'saccade_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
            if os.path.exists(fix_path):
                with open(fix_path, 'rb') as f:
                    fix_indices = pickle.load(f)
                    fixation_rows.append({'session_name': session, 'interaction_type': interaction_type, 'run_number': run, 'agent': agent, 'fixation_start_stop': fix_indices})
            if os.path.exists(sacc_path):
                with open(sacc_path, 'rb') as f:
                    sacc_indices = pickle.load(f)
                    saccade_rows.append({'session_name': session, 'interaction_type': interaction_type, 'run_number': run, 'agent': agent, 'saccade_start_stop': sacc_indices})
    else:
        for task in df_keys_for_tasks:
            session, interaction_type, run, agent, positions = task
            fixation_start_stop_inds = _detect_fixation_and_saccade_in_run(positions, session)
            fixation_rows.append({'session_name': session, 'interaction_type': interaction_type, 'run_number': run, 'agent': agent, 'fixation_start_stop': fixation_start_stop_inds})
            pdb.set_trace()
            '''
            add stuff to make dataframes in here
            '''

    return pd.DataFrame(fixation_rows), pd.DataFrame(saccade_rows)


def _detect_fixation_and_saccade_in_run(positions, session_name):
    non_nan_chunks, chunk_start_indices = __extract_non_nan_chunks(positions)
    print(f"Number of non-NaN chunks of data found: {len(chunk_start_indices)} of lengths {[chunk.shape[0] for chunk in non_nan_chunks]}")
    all_fix_start_stops = np.empty((0, 2), dtype=int) 
    for position_chunk, start_ind in zip(non_nan_chunks, chunk_start_indices):
        fixation_start_stop_indices = fixation_detector.detect_fixation_in_position_array(position_chunk, session_name)
        fixation_start_stop_indices += start_ind
        all_fix_start_stops = np.concatenate((all_fix_start_stops, fixation_start_stop_indices), axis=0)
        '''
        fixations handled, now implement the saccade part here
        '''

        # saccade_indices = saccade_detector.detect_saccade_in_position_array(position_chunk)
        # saccade_indices += start_ind
    return all_fix_start_stops



def __extract_non_nan_chunks(positions):
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
    return non_nan_chunks, start_indices



if __name__ == "__main__":
    main()
