import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count

import pdb

import sys
from pathlib import Path
# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import curate_data
import load_data

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'remake_firing_rate_df': True,
        'neural_data_bin_size': 10,  # 10 ms in seconds
        'smooth_spike_counts': True,
        'gaussian_smoothing_sigma': 5,
        'time_window_before_event': 500
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    logger.info("Parameters initialized successfully")
    return params


def main():
    logger.info("Starting the script")
    params = _initialize_params()
    sparse_nan_removed_sync_gaze_data_df_filepath = os.path.join(
        params['processed_data_dir'], 'sparse_nan_removed_sync_gaze_data_df.pkl'
    )
    eye_mvm_behav_df_file_path = os.path.join(
        params['processed_data_dir'], 'eye_mvm_behav_df.pkl'
    )
    spike_times_file_path = os.path.join(
        params['processed_data_dir'], 'spike_times_df.pkl'
    )
    behav_firing_rate_df_file_path = os.path.join(
        params['processed_data_dir'], 'behavioral_firing_rate_df.pkl'
    )
    logger.info("Loading data files")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_df = load_data.get_data_df(spike_times_file_path)

    if params.get('remake_firing_rate_df', False):
        logger.info("Computing trial-wise firing-rate dataframe")
        behav_firing_rate_df = compute_firing_rates_for_fixations_and_saccades(
            eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, params
        )
    else:
        logger.info("Loading trial-wise firing-rate dataframe")
        behav_firing_rate_df = load_data.get_data_df(behav_firing_rate_df_file_path)

    print("Firing-rate dataframe head:")
    print(behav_firing_rate_df.head())


def compute_firing_rates_for_fixations_and_saccades(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, params):
    num_cpus = min(16, cpu_count())
    output_path = os.path.join(params['processed_data_dir'], 'behavioral_firing_rate_df.pkl')
    
    session_groups = list(eye_mvm_behav_df.groupby('session_name'))
    
    with Pool(num_cpus) as pool, tqdm(total=len(session_groups), desc='Processing Sessions') as pbar:
        results = []
        for result in pool.starmap(extract_firing_rates_for_session, 
                                   [(session_name, session_group, sparse_nan_removed_sync_gaze_df, spike_times_df, params) 
                                    for session_name, session_group in session_groups]):
            results.append(result)
            pbar.update()
    
    firing_rate_data = [item for sublist in results for item in sublist]
    firing_rate_df = pd.DataFrame(firing_rate_data)
    firing_rate_df.to_pickle(output_path)

    return firing_rate_df


def extract_firing_rates_for_session(session_name, session_group, sparse_nan_removed_sync_gaze_df, spike_times_df, params):
    bin_size = params['neural_data_bin_size'] / 1000.0  # Convert to seconds
    sigma = params['gaussian_smoothing_sigma']
    time_windows = [200, 400, 600, 800, 1000]  # ms
    time_windows = [t / 1000.0 for t in time_windows]  # Convert to seconds
    firing_rate_data = []
    
    session_spikes = spike_times_df[spike_times_df['session_name'] == session_name]
    session_gaze = sparse_nan_removed_sync_gaze_df[
        (sparse_nan_removed_sync_gaze_df['session_name'] == session_name)
    ]
    
    for run_number, run_group in session_group.groupby('run_number'):
        run_gaze = session_gaze[session_gaze['run_number'] == run_number]
        if run_gaze.empty:
            continue
        
        run_gaze = run_gaze[run_gaze['agent'] == 'm1']
        run_behaviors = run_group[run_group['agent'] == 'm1']
        neural_time_list = run_gaze.iloc[0]['neural_timeline']
        
        for _, row in run_behaviors.iterrows():
            fixations = row['fixation_start_stop']
            fix_locations = row['fixation_location']
            
            categorized_fixations = [
                "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
                "non_eye_face" if "face" in set(fixes) else
                "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
                for fixes in fix_locations
            ]
            
            for (fix_start, fix_end), fix_category in zip(fixations, categorized_fixations):
                duration = (fix_end - fix_start) / 1000.0
                if duration > 1.0:
                    continue
                
                duration_category = min([t for t in time_windows if duration <= t], default=None)
                if duration_category is None or fix_start >= len(neural_time_list):
                    continue
                
                fixation_time = neural_time_list[fix_start][0]
                
                for _, neuron_row in session_spikes.iterrows():
                    bins = np.linspace(fixation_time - 0.5, fixation_time + duration_category, int((duration_category + 0.5) / bin_size) + 1).ravel()
                    firing_rate, _ = np.histogram(neuron_row['spike_ts'], bins=bins)
                    firing_rate = gaussian_filter1d(firing_rate / bin_size, sigma=sigma)
                    
                    firing_rate_data.append({
                        'session_name': session_name,
                        'behavior_type': 'fixation',
                        'location': fix_category,
                        'from_location': None,
                        'to_location': None,
                        'behav_duration_category': duration_category,
                        'run_number': run_number,
                        'unit_uuid': neuron_row['unit_uuid'],
                        'region': neuron_row['region'],
                        'firing_rate_timeline': firing_rate.tolist()
                    })
            
            saccades = row['saccade_start_stop']
            saccade_from = row['saccade_from']
            saccade_to = row['saccade_to']
            
            categorized_saccade_from = [
                "eyes" if {"face", "eyes_nf"}.issubset(set(sacc)) else
                "non_eye_face" if "face" in set(sacc) else
                "object" if set(sacc) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
                for sacc in saccade_from
            ]
            
            categorized_saccade_to = [
                "eyes" if {"face", "eyes_nf"}.issubset(set(sacc)) else
                "non_eye_face" if "face" in set(sacc) else
                "object" if set(sacc) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
                for sacc in saccade_to
            ]
            
            for (sacc_start, sacc_end), sacc_from_cat, sacc_to_cat in zip(saccades, categorized_saccade_from, categorized_saccade_to):
                duration = (sacc_end - sacc_start) / 1000.0
                if sacc_start >= len(neural_time_list):
                    continue
                
                saccade_time = neural_time_list[sacc_start][0]
                bins = np.linspace(saccade_time - 0.3, saccade_time + 0.2, int((0.2 + 0.3) / bin_size) + 1).ravel()
                
                for _, neuron_row in session_spikes.iterrows():
                    firing_rate, _ = np.histogram(neuron_row['spike_ts'], bins=bins)
                    firing_rate = gaussian_filter1d(firing_rate / bin_size, sigma=sigma)
                    
                    firing_rate_data.append({
                        'session_name': session_name,
                        'behavior_type': 'saccade',
                        'location': None,
                        'from_location': sacc_from_cat,
                        'to_location': sacc_to_cat,
                        'behav_duration_category': 0.2,
                        'run_number': run_number,
                        'unit_uuid': neuron_row['unit_uuid'],
                        'region': neuron_row['region'],
                        'firing_rate_timeline': firing_rate.tolist()
                    })
    
    return firing_rate_data



if __name__ == "__main__":
    main()



