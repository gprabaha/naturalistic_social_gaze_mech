import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, cpu_count, Manager
import itertools

import pdb

import load_data
import curate_data


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)



def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'neural_data_bin_size': 0.01,  # 10 ms in seconds
        'smooth_spike_counts': True,
        'time_window_before_and_after_event_for_psth': 0.5,
        'gaussian_smoothing_sigma': 2,
        'min_consecutive_sig_bins': 5,
        'min_total_sig_bins': 25
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
    # plot_mutual_face_fixation_density(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params)
    plot_neural_response_to_mutual_face_fixations(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, params)





def plot_mutual_face_fixation_density(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params):
    """
    Computes and plots the density of mutual face fixations across time for each session in parallel,
    while tracking progress using tqdm.

    Parameters:
    - eye_mvm_behav_df: DataFrame containing fixation data.
    - sparse_nan_removed_sync_gaze_df: DataFrame containing synchronized gaze data.
    - params: Dictionary containing root directory paths.
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    root_dir = os.path.join(params['root_data_dir'], "plots", "mutual_face_fix_density", today_date)
    os.makedirs(root_dir, exist_ok=True)

    session_names = eye_mvm_behav_df['session_name'].unique()
    num_processes = min(8, cpu_count())  # Limit parallel processes

    # Use Manager to create a shared counter for tqdm
    with Manager() as manager:
        progress = manager.Value('i', 0)  # Shared counter
        total_sessions = len(session_names)

        with tqdm(total=total_sessions, desc="Processing Sessions", position=0) as pbar:
            pool = Pool(num_processes)
            results = []

            # Run processes asynchronously to update tqdm manually
            for session_name in session_names:
                result = pool.apply_async(
                    _process_session_face_fixation_density, 
                    (session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params, root_dir, progress)
                )
                results.append(result)

            for result in results:
                result.get()  # Wait for each process to finish
                progress.value += 1  # Increment shared counter
                pbar.update(1)  # Update tqdm in main process

            pool.close()
            pool.join()

    print(f"Plots saved in {root_dir}")


def _process_session_face_fixation_density(session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, params, root_dir, progress):
    """
    Processes and plots the mutual face fixation density for a given session.
    """
    session_df = eye_mvm_behav_df[eye_mvm_behav_df['session_name'] == session_name]
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

        # Extract fixation data
        m1_fix_intervals = m1_data['fixation_start_stop'].iloc[0]
        m1_fix_locations = m1_data['fixation_location'].iloc[0]
        m2_fix_intervals = m2_data['fixation_start_stop'].iloc[0]
        m2_fix_locations = m2_data['fixation_location'].iloc[0]

        # Compute average fixation duration
        avg_fix_duration_m1 = np.mean([stop - start for start, stop in m1_fix_intervals])
        avg_fix_duration_m2 = np.mean([stop - start for start, stop in m2_fix_intervals])
        avg_fix_duration = int(np.mean([avg_fix_duration_m1, avg_fix_duration_m2]))  # Average across both agents
        window_size = max(5 * avg_fix_duration, 1)  # Ensure window size is at least 1
        sigma = 3 * window_size  # Dynamic sigma for smoothing

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

        # Compute face fixation density for M1
        for (start, stop), location in zip(m1_fix_intervals, m1_fix_locations):
            if 'face' in location:
                m1_face_density[start:stop + 1] += 1  # Mark face fixation

        # Compute face fixation density for M2
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
        ax.plot(time_axis, rolling_m1_face_density, label="M1 Face Fixation Density", color="blue", alpha=0.7)
        ax.plot(time_axis, rolling_m2_face_density, label="M2 Face Fixation Density", color="green", alpha=0.7)
        ax.plot(time_axis, mutual_density, label="Mutual Face Fixation Density (Interactive Periods)", color="red", linewidth=2)

        ax.set_title(f"Session: {session_name}, Run: {run_number} - Face Fixation Density (Window={window_size}, Sigma={sigma:.2f})")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(root_dir, f"{session_name}_face_fixation_density.png")
    plt.savefig(save_path, dpi=100)  # Set DPI to 100
    plt.close()




def plot_neural_response_to_mutual_face_fixations(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, params):
    """
    Analyzes neural response to face fixations of M1 during high vs. low mutual face fixation density periods.

    Parameters:
    - eye_mvm_behav_df: DataFrame containing fixation data.
    - sparse_nan_removed_sync_gaze_df: DataFrame containing synchronized gaze data.
    - spike_times_df: DataFrame containing spike timing information.
    - params: Dictionary containing root directory paths.
    """
    logger.info("Starting neural analysis for mutual face fixation periods")

    # Set up plot save directory
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_date += '_5-25_minbin'
    root_dir = os.path.join(params['root_data_dir'], "plots", "neural_response_mutual_face_fix", today_date)
    os.makedirs(root_dir, exist_ok=True)

    # Get unique session names for parallel processing
    session_names = eye_mvm_behav_df['session_name'].unique()
    num_processes = min(8, cpu_count())  # Use configured CPUs, fallback to 8

    # Prepare arguments for parallel execution
    task_args = [(session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, root_dir, params) for session_name in session_names]

    merged_sig_units = {}
    merged_non_sig_units = {}

    # Use tqdm with imap_unordered for better load balancing
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_process_session_wrapper, task_args), total=len(session_names), desc="Processing sessions"))

    # Merge results
    for sig_units, non_sig_units in results:
        for region, units in sig_units.items():
            merged_sig_units.setdefault(region, []).extend(units)
        for region, units in non_sig_units.items():
            merged_non_sig_units.setdefault(region, []).extend(units)

    # Compute summary counts
    summary_counts = {
        region: (len(merged_sig_units.get(region, [])), len(merged_sig_units.get(region, [])) + len(merged_non_sig_units.get(region, [])))
        for region in set(merged_sig_units.keys()).union(set(merged_non_sig_units.keys()))
    }

    # Display results
    logger.info("Significant neuron counts by brain region:")
    for region, (sig_count, total_count) in summary_counts.items():
        logger.info(f"{region}: {sig_count} / {total_count} significant units")


def _process_session_wrapper(args):
    """ Wrapper function to unpack arguments for multiprocessing """
    return _process_session(*args)



def _process_session(session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, root_dir, params):
    """
    Processes a single session, computing and plotting neural response for mutual face fixation periods.
    """
    logger.debug(f'Processing session {session_name}')
    min_consecutive_sig_bins = params.get('min_consecutive_sig_bins', 5)
    min_total_sig_bins = params.get('min_total_sig_bins', 25)
    high_density_fixations = []
    low_density_fixations = []
    session_df = eye_mvm_behav_df[eye_mvm_behav_df['session_name'] == session_name]
    for run_number in session_df['run_number'].unique():
        run_df = session_df[session_df['run_number'] == run_number]
        m1_data = run_df[run_df['agent'] == 'm1']
        m2_data = run_df[run_df['agent'] == 'm2']
        if m1_data.empty or m2_data.empty:
            continue  # Skip if either agent is missing
        # Extract fixation data
        m1_fix_intervals = m1_data['fixation_start_stop'].iloc[0]
        m1_fix_locations = m1_data['fixation_location'].iloc[0]
        m2_fix_intervals = m2_data['fixation_start_stop'].iloc[0]
        m2_fix_locations = m2_data['fixation_location'].iloc[0]
        # Extract gaze data
        gaze_data = sparse_nan_removed_sync_gaze_df[
            (sparse_nan_removed_sync_gaze_df['session_name'] == session_name) & 
            (sparse_nan_removed_sync_gaze_df['run_number'] == run_number) &
            (sparse_nan_removed_sync_gaze_df['agent'] == 'm1')
        ]
        if gaze_data.empty:
            continue
        # Get neural timeline
        neural_timeline = np.array(gaze_data['neural_timeline'].iloc[0]).flatten()
        timeline_len = len(neural_timeline)
        # Compute fixation durations
        avg_fix_duration_m1 = np.mean([stop - start for start, stop in m1_fix_intervals])
        avg_fix_duration_m2 = np.mean([stop - start for start, stop in m2_fix_intervals])
        avg_fix_duration = int(np.mean([avg_fix_duration_m1, avg_fix_duration_m2]))
        window_size = max(5 * avg_fix_duration, 1)
        sigma = 3 * window_size  # Dynamic sigma for smoothing
        # Initialize face fixation density arrays
        m1_face_density = np.zeros(timeline_len)
        m2_face_density = np.zeros(timeline_len)
        # Compute face fixation density for M1
        for (start, stop), location in zip(m1_fix_intervals, m1_fix_locations):
            if 'face' in location:
                m1_face_density[start:stop + 1] += 1  # Mark face fixation
        # Compute face fixation density for M2
        for (start, stop), location in zip(m2_fix_intervals, m2_fix_locations):
            if 'face' in location:
                m2_face_density[start:stop + 1] += 1  # Mark face fixation
        # Compute mutual fixation density
        rolling_m1_face_density = np.convolve(m1_face_density, np.ones(window_size), mode='same') / window_size
        rolling_m2_face_density = np.convolve(m2_face_density, np.ones(window_size), mode='same') / window_size
        smoothed_m1 = gaussian_filter1d(rolling_m1_face_density, sigma=sigma)
        smoothed_m2 = gaussian_filter1d(rolling_m2_face_density, sigma=sigma)
        mutual_density = np.sqrt(smoothed_m1 * smoothed_m2)
        # Threshold mutual density to define high/low density fixations
        mutual_density_threshold = np.mean(mutual_density)*0.63
        # Identify face fixations of M1 in high vs. low mutual density periods
        for (start, stop), location in zip(m1_fix_intervals, m1_fix_locations):
            if 'face' in location:
                fixation_time_in_neurons = neural_timeline[start]  # Convert to neural time
                if mutual_density[start] > mutual_density_threshold:
                    high_density_fixations.append(fixation_time_in_neurons)
                else:
                    low_density_fixations.append(fixation_time_in_neurons)
    if not high_density_fixations and not low_density_fixations:
        return  # Skip session if no valid fixations found
    # Load spike data for the session
    session_spike_data = spike_times_df[spike_times_df['session_name'] == session_name]
    if session_spike_data.empty:
        return
    # Process each neuron separately
    sig_units = {}
    non_sig_units = {}
    for _, unit in session_spike_data.iterrows():
        unit_uuid = unit["unit_uuid"]
        brain_region = unit["region"]
        logger.debug(f'Plotting for {unit_uuid} in {brain_region}')
        spike_times = np.array(unit["spike_ts"])
        is_sig = 0
        # Compute spike counts per trial
        high_density_spike_counts, timeline = _compute_spiking_per_trial(high_density_fixations, spike_times, params)
        low_density_spike_counts, _ = _compute_spiking_per_trial(low_density_fixations, spike_times, params)
        # Perform t-test per bin with FDR correction
        significant_bins = _perform_ttest_and_fdr_correct(high_density_spike_counts, low_density_spike_counts, timeline)
        groups = np.split(significant_bins, np.where(np.diff(significant_bins) > 1)[0] + 1)
        longest_run = max(len(g) for g in groups)  # Ensure last run is considered
        if (longest_run >= min_consecutive_sig_bins) or (len(significant_bins) >= min_total_sig_bins):
            is_sig = 1
            sig_units.setdefault(brain_region, []).append(unit_uuid)
        else:
            non_sig_units.setdefault(brain_region, []).append(unit_uuid)
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timeline[:-1], np.mean(high_density_spike_counts, axis=0), label="High Mutual Density", color="blue")
        ax.plot(timeline[:-1], np.mean(low_density_spike_counts, axis=0), label="Low Mutual Density", color="green")
        for bin_idx in significant_bins:
            ax.axvline(timeline[bin_idx], color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"Neural Response to Face Fixations - {unit_uuid} ({brain_region})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.legend()
        save_dir = os.path.join(root_dir, "sig_units") if is_sig else root_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{session_name}_{brain_region}_{unit_uuid}_face_fix.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
    return sig_units, non_sig_units


def _compute_spiking_per_trial(fixation_times, spike_times, params):
    """Computes spike counts per fixation trial."""
    spike_counts_per_trial = []
    bin_size = params["neural_data_bin_size"]
    sigma = params["gaussian_smoothing_sigma"]
    time_window = params['time_window_before_and_after_event_for_psth']
    timeline = np.arange(-time_window, time_window + bin_size, bin_size)
    for fixation_time in fixation_times:
        bins = np.linspace(
            fixation_time - time_window,
            fixation_time + time_window,
            int(2 * time_window / bin_size) + 1
        ).ravel()
        spike_counts, _ = np.histogram(spike_times, bins=bins)
        spike_counts = spike_counts / bin_size  # Convert to firing rate
        if params["smooth_spike_counts"]:
            spike_counts = gaussian_filter1d(spike_counts, sigma=sigma)
        spike_counts_per_trial.append(spike_counts)
    return np.array(spike_counts_per_trial), timeline


def _perform_ttest_and_fdr_correct(high_density_spike_counts, low_density_spike_counts, timeline, fdr_correct=False):
    """Performs t-test per bin and applies FDR correction."""
    num_bins = len(timeline) - 1
    p_values = []
    # Define the time window for counting significant bins
    time_window_start, time_window_end = -0.45, 0.45
    valid_bins = (timeline[:-1] >= time_window_start) & (timeline[:-1] <= time_window_end)
    for bin_idx in range(num_bins):
        p_value = ttest_ind(
            high_density_spike_counts[:, bin_idx],
            low_density_spike_counts[:, bin_idx],
            equal_var=False
        )[1]
        p_values.append(p_value)
    # Apply FDR correction
    if fdr_correct:
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    else:
        corrected_p_values = np.array(p_values)
    # Identify significant bins
    significant_bins = np.where((corrected_p_values < 0.05) & (valid_bins))[0]
    return significant_bins



if __name__ == "__main__":
    main()