import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d


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
    logger.info("Loading data files")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_df = load_data.get_data_df(spike_times_file_path)

    preprocessed_dataframe = compute_firing_rates_for_fixations_and_saccades(
        eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, params
    )

    pdb.set_trace()

    return 0


def compute_firing_rates_for_fixations_and_saccades(
    eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, params
):
    bin_size = params['neural_data_bin_size'] / 1000.0  # Convert to seconds
    sigma = params['gaussian_smoothing_sigma']
    time_windows = [200, 400, 600, 800, 1000]  # ms
    time_windows = [t / 1000.0 for t in time_windows]  # Convert to seconds
    behavioral_data = []    
    for session_name, session_group in tqdm(eye_mvm_behav_df.groupby('session_name'), desc="Preprocessing sessions"):
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
            for _, row in run_behaviors.iterrows():
                fixations = row['fixation_start_stop']
                fix_locations = row['fixation_location']
                # Categorize fixation locations
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
                    # Determine duration category
                    duration_category = min([t for t in time_windows if duration <= t], default=None)
                    if duration_category is None:
                        continue
                    # Get fixation start time in neural timeline
                    neural_time_list = run_gaze.iloc[0]['neural_timeline']
                    if fix_start >= len(neural_time_list):
                        continue
                    fixation_time = neural_time_list[fix_start][0]
                    for _, neuron_row in session_spikes.iterrows():
                        unit_uuid = neuron_row['unit_uuid']
                        region = neuron_row['region']
                        spike_times = np.array(neuron_row['spike_ts'])
                        # Extract spike counts in bins
                        bins = np.linspace(fixation_time - 0.5, fixation_time + duration_category, int((duration_category + 0.5) / bin_size) + 1).ravel()
                        firing_rate, _ = np.histogram(spike_times, bins=bins)
                        firing_rate = firing_rate / bin_size  # Normalize by bin size
                        firing_rate = gaussian_filter1d(firing_rate, sigma=sigma)  # Apply Gaussian smoothing
                        behavioral_data.append({
                            'session_name': session_name,
                            'behavior_type': 'fixation',
                            'location': fix_category,
                            'from_location': None,
                            'to_location': None,
                            'behav_duration_category': duration_category,
                            'run_number': run_number,
                            'unit_uuid': unit_uuid,
                            'region': region,
                            'firing_rate_timeline': firing_rate.tolist()
                        })
                # Process saccades
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
                    # Saccades have a fixed duration category of 0.2s (200ms)
                    duration_category = 0.2
                    # Get saccade start time in neural timeline
                    if sacc_start >= len(neural_time_list):
                        continue
                    saccade_time = neural_time_list[sacc_start][0]
                    for _, neuron_row in session_spikes.iterrows():
                        unit_uuid = neuron_row['unit_uuid']
                        region = neuron_row['region']
                        spike_times = np.array(neuron_row['spike_ts'])
                        # Extract spike counts in bins (300 ms before start to 200 ms after)
                        bins = np.linspace(saccade_time - 0.3, saccade_time + duration_category, int((duration_category + 0.3) / bin_size) + 1).ravel()
                        firing_rate, _ = np.histogram(spike_times, bins=bins)
                        firing_rate = firing_rate / bin_size  # Normalize by bin size
                        firing_rate = gaussian_filter1d(firing_rate, sigma=sigma)  # Apply Gaussian smoothing
                        behavioral_data.append({
                            'session_name': session_name,
                            'behavior_type': 'saccade',
                            'location': None,
                            'from_location': sacc_from_cat,
                            'to_location': sacc_to_cat,
                            'behav_duration_category': duration_category,
                            'run_number': run_number,
                            'unit_uuid': unit_uuid,
                            'region': region,
                            'firing_rate_timeline': firing_rate.tolist()
                        })
    return pd.DataFrame(behavioral_data)



if __name__ == "__main__":
    main()



