import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

import pdb

import load_data
import curate_data


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


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
    _plot_mutual_face_fixation_density(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params)

def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'neural_data_bin_size': 0.01,  # 10 ms in seconds
        'smooth_spike_counts': True,
        'time_window_before_and_after_event_for_psth': 0.5
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    logger.info("Parameters initialized successfully")
    return params



def _plot_mutual_face_fixation_density(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params):
    """
    Computes and plots the density of mutual face fixations across time for each session and run.

    Parameters:
    - eye_mvm_behav_df: DataFrame containing fixation data.
    - sparse_nan_removed_sync_gaze_df: DataFrame containing synchronized gaze data.
    - params: Dictionary containing root directory paths.
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    root_dir = os.path.join(params['root_data_dir'], "plots", "mutual_face_fix_density", today_date)
    os.makedirs(root_dir, exist_ok=True)

    session_groups = eye_mvm_behav_df.groupby(['session_name'])

    for session_name, session_df in tqdm(session_groups, desc="Processing sessions"):
        runs = session_df['run_number'].unique()
        num_runs = len(runs)
        
        fig, axes = plt.subplots(num_runs, 1, figsize=(12, 4 * num_runs), sharex=True)
        if num_runs == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one run.

        for ax, run_number in zip(axes, runs):
            run_df = session_df[session_df['run_number'] == run_number]
            m1_data = run_df[run_df['agent'] == 'm1']
            m2_data = run_df[run_df['agent'] == 'm2']

            if m1_data.empty or m2_data.empty:
                continue  # Skip if either agent is missing

            # Extract fixation data (keeping them as lists)
            m1_fix_intervals = m1_data['fixation_start_stop'].iloc[0]
            m1_fix_locations = m1_data['fixation_location'].iloc[0]

            m2_fix_intervals = m2_data['fixation_start_stop'].iloc[0]
            m2_fix_locations = m2_data['fixation_location'].iloc[0]

            # Compute average fixation duration
            avg_fix_duration_m1 = np.mean([stop - start for start, stop in m1_fix_intervals])
            avg_fix_duration_m2 = np.mean([stop - start for start, stop in m2_fix_intervals])
            avg_fix_duration = int(np.mean([avg_fix_duration_m1, avg_fix_duration_m2]))  # Average across both agents
            window_size = max(5 * avg_fix_duration, 1)  # Ensure window size is at least 1
            sigma = window_size  # Dynamic sigma for smoothing

            # Extract number of samples from gaze data
            gaze_data = sparse_nan_removed_sync_gaze_df[
                (sparse_nan_removed_sync_gaze_df['session_name'] == session_name) & 
                (sparse_nan_removed_sync_gaze_df['run_number'] == run_number)
            ]
            if gaze_data.empty:
                continue

            timeline_len = len(gaze_data['positions'].iloc[0])  # Number of samples in the run

            # Initialize fixation density arrays
            m1_face_density = np.zeros(timeline_len)
            m2_face_density = np.zeros(timeline_len)

            # Compute face fixation density for m1
            for (start, stop), location in zip(m1_fix_intervals, m1_fix_locations):
                if 'face' in location:
                    m1_face_density[start:stop + 1] += 1  # Mark face fixation

            # Compute face fixation density for m2
            for (start, stop), location in zip(m2_fix_intervals, m2_fix_locations):
                if 'face' in location:
                    m2_face_density[start:stop + 1] += 1  # Mark face fixation

            # Convert absolute counts to relative densities using dynamic window size
            rolling_m1_face_density = np.convolve(m1_face_density, np.ones(window_size), mode='same') / window_size
            rolling_m2_face_density = np.convolve(m2_face_density, np.ones(window_size), mode='same') / window_size

            # Apply Gaussian smoothing to approximate interactiveness
            smoothed_m1 = gaussian_filter1d(rolling_m1_face_density, sigma=sigma)
            smoothed_m2 = gaussian_filter1d(rolling_m2_face_density, sigma=sigma)

            # Compute mutual density using a time-proximity weighting instead of direct multiplication
            mutual_density = np.sqrt(smoothed_m1 * smoothed_m2)  # Geometric mean to balance both densities

            # Create relative time axis in minutes
            time_axis = np.arange(timeline_len) / (1000 * 60)  # Convert samples to minutes

            # Plot results
            ax.plot(time_axis, smoothed_m1, label="M1 Face Fixation Density", color="blue", alpha=0.7)
            ax.plot(time_axis, smoothed_m2, label="M2 Face Fixation Density", color="green", alpha=0.7)
            ax.plot(time_axis, mutual_density, label="Mutual Face Fixation Density (Interactive Periods)", color="red", linewidth=2)

            ax.set_title(f"Session: {session_name}, Run: {run_number} - Face Fixation Density (Window={window_size}, Sigma={sigma:.2f})")
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        save_path = os.path.join(root_dir, f"{session_name}_face_fixation_density.png")
        plt.savefig(save_path, dpi=100)  # Set DPI to 100
        plt.close()

    print(f"Plots saved in {root_dir}")


if __name__ == "__main__":
    main()