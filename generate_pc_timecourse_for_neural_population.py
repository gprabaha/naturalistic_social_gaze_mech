import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False

import pdb

import load_data
import curate_data
import util


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'is_cluster': True,
        'prabaha_local': True,
        'neural_data_bin_size': 0.01,  # 10 ms in seconds
        'smooth_spike_counts': True,
        'time_window_before_event_for_psth': 0.5,
        'time_window_after_event_for_psth': 1.0,
        'gaussian_smoothing_sigma': 2,
        'min_consecutive_sig_bins': 5,
        'min_total_sig_bins': 25
    }
    params = curate_data.add_num_cpus_to_params(params)
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    params = util.get_slurm_cpus_and_threads(params)
    logger.info("Parameters initialized successfully")
    return params


def main():
    logger.info("Starting the script")
    params = _initialize_params()
    processed_data_dir = params.get('processed_data_dir', './processed_data')
    sparse_nan_removed_sync_gaze_data_df_filepath = os.path.join(
        processed_data_dir, 'sparse_nan_removed_sync_gaze_data_df.pkl'
    )
    eye_mvm_behav_df_file_path = os.path.join(
        processed_data_dir, 'eye_mvm_behav_df.pkl'
    )
    spike_times_file_path = os.path.join(
        processed_data_dir, 'spike_times_df.pkl'
    )
    logger.info("Loading data files")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_df = load_data.get_data_df(spike_times_file_path)

    # Perform the merge
    eye_mvm_behav_df_with_neural_timeline = eye_mvm_behav_df.merge(
        sparse_nan_removed_sync_gaze_df[['session_name', 'interaction_type', 'run_number', 'agent', 'positions', 'neural_timeline']],
        on=['session_name', 'interaction_type', 'run_number', 'agent'],
        how='left'  # Use 'left' join to preserve all rows in eye_mvm_behav_df
    )
    del sparse_nan_removed_sync_gaze_df, eye_mvm_behav_df
    gc.collect()

    fixation_firing_rate_df = compute_firing_rate_matrix(eye_mvm_behav_df_with_neural_timeline, spike_times_df, params)

    pdb.set_trace()


    logger.info("Script finished running!")




def compute_firing_rate_matrix(eye_mvm_behav_df, spike_times_df, params):
    """
    Computes the trial-by-trial, binned firing rate timecourse matrix for each unit in each session and region.
    
    Parameters:
    - eye_mvm_behav_df: DataFrame containing fixation events with neural timeline and positions.
    - spike_times_df: DataFrame containing spike timestamps for each unit.
    - params: Dictionary containing:
        - 'neural_data_bin_size': Bin size for firing rate calculation (seconds).
        - 'gaussian_smoothing_sigma': Standard deviation for Gaussian smoothing.
        - 'time_window_before_event': Time before fixation start to include in PSTH (seconds).
        - 'time_window_after_event': Time after fixation start to include in PSTH (seconds).

    Returns:
    - firing_rate_df: DataFrame with columns ['session_name', 'region', 'unit_uuid', 'fixation_type', 'firing_rate_matrix']
    """
    bin_size = params.get('neural_data_bin_size', 0.01)
    smooth_sigma = params.get('gaussian_smoothing_sigma', 2)
    time_window_before = params.get('time_window_before_event_for_psth', 0.5)  # Fixed at 0.5 seconds before
    time_window_after = params.get('time_window_after_event_for_psth', 1.0)   # Fixed at 1.0 second after

    results = []

    # Iterate over sessions
    for session, session_spike_df in spike_times_df.groupby('session_name'):
        # Filter fixation data for this session and agent 'm1'
        session_m1_fixations = eye_mvm_behav_df[
            (eye_mvm_behav_df['session_name'] == session) & 
            (eye_mvm_behav_df['agent'] == 'm1')
        ]

        # Iterate over runs
        for run, run_fixations in session_m1_fixations.groupby('run_number'):
            # Identify fixation types (each fixation gets a single type)

            def get_fixation_type(fix_locs):
                """
                Returns a LIST of all matching fixation types instead of a single label.
                """
                fix_type = []
                for sublist in fix_locs:
                    appended = 0
                    for loc in sublist:
                        if 'face' in loc or 'mouth' in loc:
                            fix_type.append('face')
                            appended = 1
                            break
                        if 'object' in loc:
                            fix_type.append('object')
                            appended = 1
                            break
                    if appended == 0:
                        fix_type.append('out_of_roi')
                return fix_type  # Return list
            
            run_fixations['fixation_type'] = run_fixations['fixation_location'].apply(get_fixation_type)
            # Drop fixations with no valid type
            run_fixations = run_fixations.dropna(subset=['fixation_type'])
            # Iterate over spike data for the current session
            for _, unit_row in session_spike_df.iterrows():
                unit_uuid = unit_row['unit_uuid']
                region = unit_row['region']
                spike_times = np.array(unit_row['spike_ts'])  # Convert spike times to NumPy array
                pdb.set_trace()
                for fixation_type, fix_group in run_fixations.groupby('fixation_type'):
                    firing_rate_list = []

                    for _, fixation_row in fix_group.iterrows():
                        neural_times = np.array(fixation_row['neural_timeline']).flatten()
                        fixation_start = neural_times[0]  # Start of fixation
                        start_time = fixation_start - time_window_before
                        end_time = fixation_start + time_window_after

                        # Define bins for spike counts
                        bins = np.arange(start_time, end_time, bin_size).ravel()
                        binned_spike_counts, _ = np.histogram(spike_times, bins=bins)

                        # Gaussian smoothing
                        smoothed_firing_rate = gaussian_filter1d(binned_spike_counts / bin_size, sigma=smooth_sigma)
                        firing_rate_list.append(smoothed_firing_rate)

                    # Store results
                    if firing_rate_list:
                        firing_rate_matrix = np.vstack(firing_rate_list)
                        results.append({
                            'session_name': session,
                            'run_number': run,
                            'region': region,
                            'unit_uuid': unit_uuid,
                            'fixation_type': fixation_type,
                            'firing_rate_matrix': firing_rate_matrix
                        })

    # Convert to DataFrame
    firing_rate_df = pd.DataFrame(results)
    return firing_rate_df






if __name__ == "__main__":
    main()

