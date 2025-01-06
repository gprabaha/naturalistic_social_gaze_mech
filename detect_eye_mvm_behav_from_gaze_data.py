
import os
import random
import pandas as pd
import pickle
import numpy as np
from scipy.interpolate import interp1d

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


def _prepare_tasks(synchronized_gaze_data_df, params):
    df_keys_for_tasks = synchronized_gaze_data_df[
        ['session_name', 'interaction_type', 'run_number', 'agent', 'positions']
    ].values.tolist()
    if params.get('try_using_single_run'):
        random_run = random.choice(df_keys_for_tasks)
        random_session, random_interaction_type, random_run_num, random_agent, _ = random_run
        df_keys_for_tasks = [random_run]
        print(f"!! Testing using positions data from a random single run: {random_session}, {random_interaction_type}, {random_run_num}, {random_agent}!!")
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
            _detect_fixation_and_saccade_in_run(positions, session)
            '''
            add stuff to make dataframes in here
            '''

    return pd.DataFrame(fixation_rows), pd.DataFrame(saccade_rows)


def _detect_fixation_and_saccade_in_run(positions, session_name):
    pdb.set_trace()
    positions = __interpolate_nans_in_positions_with_sliding_window(positions)
    pdb.set_trace()
    non_nan_chunks, chunk_start_indices = __extract_non_nan_chunks(positions)
    pdb.set_trace()
    for position_chunk, start_ind in zip(non_nan_chunks, chunk_start_indices):
        fixation_indices = fixation_detector.detect_fixation_in_position_array(position_chunk, session_name)
        fixation_indices += start_ind
        # saccade_indices = saccade_detector.detect_saccade_in_position_array(position_chunk)
        # saccade_indices += start_ind


def __interpolate_nans_in_positions_with_sliding_window(positions, window_size=15, max_nans=5):
    positions = positions.copy()
    num_points = positions.shape[0]
    for start in range(num_points - window_size + 1):
        end = start + window_size
        window = positions[start:end]
        nan_mask = np.isnan(window).any(axis=1)
        nan_count = np.sum(nan_mask)
        if nan_count <= max_nans:
            for col in range(positions.shape[1]):
                valid_indices = np.where(~np.isnan(window[:, col]))[0]
                if len(valid_indices) > 1:
                    valid_values = window[valid_indices, col]
                    interp_func = interp1d(valid_indices, valid_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    nan_indices = np.where(nan_mask)[0]
                    interpolated_values = interp_func(nan_indices)
                    window[nan_indices, col] = interpolated_values
            positions[start:end] = window
    return positions


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
