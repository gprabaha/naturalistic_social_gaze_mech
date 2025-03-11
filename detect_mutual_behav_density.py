import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, cpu_count, Manager
from joblib import Parallel, delayed, parallel_backend
from functools import partial

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
        'parallelize_over_sessions': False,
        'recalculate_mutual_density_df': False,
        'fixation_type_to_process': 'face',
        'neural_data_bin_size': 0.01,  # 10 ms in seconds
        'smooth_spike_counts': True,
        'time_window_before_and_after_event_for_psth': 0.5,
        'gaussian_smoothing_sigma': 2,
        'min_consecutive_sig_bins': 5,
        'min_total_sig_bins': 25
    }
    params = curate_data.add_num_cpus_to_params(params)
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    params = get_slurm_cpus_and_threads(params)
    logger.info("Parameters initialized successfully")
    return params


def get_slurm_cpus_and_threads(params):
    """Returns the number of allocated CPUs and dynamically adjusts threads per CPU based on SLURM settings or local multiprocessing."""
    if params.get("is_cluster", False):
        # Get number of CPUs allocated by SLURM
        available_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        available_cpus = int(available_cpus) if available_cpus else 1  # Default to 1 if not in SLURM
    else:
        # Get number of available CPUs using multiprocessing
        available_cpus = cpu_count()
    # Default to 4 threads per CPU unless num_cpus is less than 4
    threads_per_cpu = 4 if available_cpus >= 4 else 1
    # Compute num_cpus by dividing total CPUs by threads per CPU
    num_cpus = max(1, available_cpus // threads_per_cpu)  # Ensure at least 1 CPU
    params['available_cpus'] = available_cpus
    params['num_cpus'] = num_cpus
    params['threads_per_cpu'] = threads_per_cpu
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
    fix_binary_vector_file = os.path.join(
        processed_data_dir, 'fix_binary_vector_df.pkl'
    )
    logger.info("Loading data files")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_df = load_data.get_data_df(spike_times_file_path)
    fix_binary_vector_df = load_data.get_data_df(fix_binary_vector_file)

    # Check if density df recalculation is needed
    fixation_type = params.get('fixation_type_to_process', 'face')
    mutual_density_filename = f"mutual_fixation_density_{fixation_type}.pkl"
    mutual_density_file_path = os.path.join(processed_data_dir, mutual_density_filename)
    if params.get('recalculate_mutual_density_df', False) or not os.path.exists(mutual_density_file_path):
        mutual_behav_density_df = detect_mutual_face_fixation_density(fix_binary_vector_df, params)
    else:
        logger.info(f"Loading precalculated mutual fixation density data from {mutual_density_file_path}")
        mutual_behav_density_df = load_data.get_data_df(mutual_density_file_path)
    
    # logger.info("Plotting fixation and density timeline of 10 random runs")
    # plot_fixation_densities_in_10_random_runs(fix_binary_vector_df, mutual_behav_density_df, params)

    merged_sig_units, merged_non_sig_units = analyze_and_plot_neural_response_to_face_fixations_during_mutual_bouts(
        eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, 
        mutual_behav_density_df, params)

    pdb.set_trace()

    logger.info("Script finished running!")




def detect_mutual_face_fixation_density(fix_binary_vector_df, params):
    """
    Detects mutual face fixation density across time for each session and saves the result as a pickle file.
    Parameters:
    - fix_binary_vector_df: DataFrame containing synchronized binary fixation data.
    - params: Dictionary containing parameters, including 'processed_data_dir'.
    Returns:
    - DataFrame with smoothed and normalized mutual face fixation densities for m1 and m2.
    """
    fixation_type = params.get('fixation_type_to_process', 'face')
    processed_data_dir = params.get('processed_data_dir', './processed_data')  # Default to local dir if missing
    logger.info(f"Starting mutual {fixation_type} fixation density detection")
    # Ensure output directory exists
    os.makedirs(processed_data_dir, exist_ok=True)
    session_groups = fix_binary_vector_df.groupby('session_name')
    n_jobs = params.get('available_cpus', 1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_fixation_density_in_one_session)(session_name, session_group, fixation_type)
        for session_name, session_group in tqdm(session_groups, desc="Processing Sessions")
    )
    # Flatten results since `get_fixation_density_in_one_session` returns lists of dictionaries
    all_run_data = [entry for session_results in results for entry in session_results]
    # Convert to DataFrame
    density_df = pd.DataFrame(all_run_data)
    # Construct output file path
    output_filename = f"mutual_fixation_density_{fixation_type}.pkl"
    output_path = os.path.join(processed_data_dir, output_filename)
    # Save DataFrame as a pickle file
    density_df.to_pickle(output_path)
    logger.info(f"Saved mutual fixation density data to {output_path}")
    return density_df

def get_fixation_density_in_one_session(session_name, session_group, fixation_type):
    """Processes all runs in a session sequentially."""

    def normalize_density(density):
        """Normalize a density array to [0,1] range."""
        return (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-8)

    session_results = []
    run_groups = session_group.groupby('run_number')
    for run_number, run_group in tqdm(run_groups, desc=f"Processing Runs in {session_name}"):
    # for run_number, run_group in run_groups:
        m1_fixations = run_group[(run_group['agent'] == 'm1') & (run_group['fixation_type'] == fixation_type)]
        m2_fixations = run_group[(run_group['agent'] == 'm2') & (run_group['fixation_type'] == fixation_type)]
        if m1_fixations.empty or m2_fixations.empty:
            continue  # Skip runs with missing data
        m1_binary_vector = np.array(m1_fixations['binary_vector'].values[0])
        m2_binary_vector = np.array(m2_fixations['binary_vector'].values[0])
        # Compute mean fixation duration & IFI for sigma estimation (separately for M1 and M2)
        m1_fix_dur, m1_ifi = compute_fixation_metrics(m1_binary_vector)
        m2_fix_dur, m2_ifi = compute_fixation_metrics(m2_binary_vector)
        # Use separate smoothing sigmas
        m1_sigma = (m1_fix_dur + m1_ifi) / 2
        m2_sigma = (m2_fix_dur + m2_ifi) / 2
        # Ensure equal length
        min_length = min(len(m1_binary_vector), len(m2_binary_vector))
        m1_binary_vector = m1_binary_vector[:min_length]
        m2_binary_vector = m2_binary_vector[:min_length]
        # Apply Gaussian smoothing with separate sigmas
        m1_density = gaussian_filter1d(m1_binary_vector.astype(float), sigma=m1_sigma, mode='constant')
        m2_density = gaussian_filter1d(m2_binary_vector.astype(float), sigma=m2_sigma, mode='constant')
        # Normalize to [0,1]
        m1_density_norm = normalize_density(m1_density)
        m2_density_norm = normalize_density(m2_density)
        # Compute mutual fixation density
        mutual_density = np.sqrt(m1_density_norm * m2_density_norm)
        # Normalize mutual fixation density
        mutual_density_norm = normalize_density(mutual_density)
        # Store results
        session_results.append({
            'session_name': session_name,
            'run_number': run_number,
            'fixation_type': fixation_type,
            'm1_fix_dur': m1_fix_dur,
            'm2_fix_dur': m2_fix_dur,
            'm1_ifi': m1_ifi,
            'm2_ifi': m2_ifi,
            'm1_sigma': m1_sigma,
            'm2_sigma': m2_sigma,
            'm1_density': list(m1_density_norm),
            'm2_density': list(m2_density_norm),
            'mutual_density': list(mutual_density_norm)
        })
    return session_results

def compute_fixation_metrics(binary_vector):
    """ 
    Computes mean fixation duration and inter-fixation interval from a binary vector.
    """
    binary_vector = np.array(binary_vector)
    if np.all(binary_vector == 0):  # No fixations at all
        return 0, 0
    if np.all(binary_vector == 1):  # Continuous fixation, no IFI
        return len(binary_vector), 0
    # Detect change points (fixation starts/stops)
    change_indices = np.where(np.diff(np.pad(binary_vector, (1, 1), 'constant')) != 0)[0]
    # Compute durations of alternating segments
    durations = np.diff(change_indices)
    # Extract fixation and IFI durations directly (fixations are at even indices)
    fix_durations = durations[::2]
    ifi_durations = durations[1::2] if len(durations) > 1 else [0]
    # Compute mean values
    mean_fix_dur = np.mean(fix_durations) if len(fix_durations) > 0 else 0
    mean_ifi = np.mean(ifi_durations) if len(ifi_durations) > 0 else 0
    return mean_fix_dur, mean_ifi



def plot_fixation_densities_in_10_random_runs(fix_binary_vector_df, mutual_behav_density_df, params):
    """
    Plots fixation durations for 10 random runs using broken_barh and overlays
    M1, M2, and mutual face fixation density estimates.
    """
    # Randomly select 10 unique runs
    selected_runs = fix_binary_vector_df[['session_name', 'run_number']].drop_duplicates().sample(n=10, random_state=42)
    # Retrieve both M1 and M2 data for each sampled run
    selected_data = fix_binary_vector_df.merge(selected_runs, on=['session_name', 'run_number'])
    # Set up the figure
    fig, axes = plt.subplots(5, 2, figsize=(8, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, (run_id, run_group) in enumerate(selected_data.groupby(['session_name', 'run_number'])):
        session_name, run_number = run_id
        # Find corresponding mutual density data
        mutual_row = mutual_behav_density_df[
            (mutual_behav_density_df['session_name'] == session_name) &
            (mutual_behav_density_df['run_number'] == run_number)
        ]
        if mutual_row.empty:
            continue
        # Extract fixation data for m1 and m2
        m1_data = run_group[(run_group['agent'] == 'm1') & (run_group['fixation_type']=="face")]
        m2_data = run_group[(run_group['agent'] == 'm2') & (run_group['fixation_type']=="face")]
        if m1_data.empty or m2_data.empty:
            continue
        # Compute fixation durations
        m1_fix_starts, m1_fix_durations = compute_fixation_durations(m1_data['binary_vector'].values[0])
        m2_fix_starts, m2_fix_durations = compute_fixation_durations(m2_data['binary_vector'].values[0])
        # Get densities
        m1_density = np.array(mutual_row['m1_density'].values[0])
        m2_density = np.array(mutual_row['m2_density'].values[0])
        mutual_density = np.array(mutual_row['mutual_density'].values[0])
        # Plot fixation durations using broken_barh for m1 and m2
        axes[i].broken_barh(list(zip(m1_fix_starts, m1_fix_durations)), (1.2, 0.4), facecolors='blue', alpha=1, label="M1 Fixations")
        axes[i].broken_barh(list(zip(m2_fix_starts, m2_fix_durations)), (0.8, 0.4), facecolors='orange', alpha=1, label="M2 Fixations")
        # Overlay densities
        time_series = np.linspace(0, len(m1_density), len(m1_density))
        axes[i].plot(time_series, m1_density, label="M1 Density", color='blue', alpha=0.7)
        axes[i].plot(time_series, m2_density, label="M2 Density", color='red', alpha=0.7)
        axes[i].plot(time_series, mutual_density, label="Mutual Density", color='green', alpha=0.9, linewidth=2.5)  # Increased linewidth
        # Titles & Labels
        axes[i].set_title(f"Session {session_name} - Run {run_number}")
        axes[i].set_ylabel("Fixation & Density")
        axes[i].set_xlabel("Time")
        # axes[i].legend(loc='upper right', fontsize=8)
    plt.suptitle("Fixation Durations with Overlaid Densities", fontsize=16)
    plt.tight_layout()
    plt.show()
    today_date = datetime.today().strftime('%Y-%m-%d')
    root_dir = os.path.join(params['root_data_dir'], "plots", "mutual_face_fix_density", today_date)
    os.makedirs(root_dir, exist_ok=True)
    save_path = os.path.join(root_dir, f"face_fixation_density_random_runs.png")
    plt.savefig(save_path, dpi=200)  # Set DPI to 200
    plt.close()


def compute_fixation_durations(binary_vector):
    """
    Computes fixation durations (consecutive 1s) and their start times from a binary vector.
    """
    binary_vector = np.array(binary_vector)
    if np.all(binary_vector == 0):  # No fixations at all
        return [], []
    if np.all(binary_vector == 1):  # Continuous fixation
        return [0], [len(binary_vector)]
    # Detect change points (fixation starts/stops)
    change_indices = np.where(np.diff(np.pad(binary_vector, (1, 1), 'constant')) != 0)[0]
    # Compute segment durations
    durations = np.diff(change_indices)
    # Fixation starts are at every alternate index (0, 2, 4, ...)
    fix_starts = change_indices[:-1:2]
    fix_durations = durations[::2]  # Fixation durations at even indices
    return fix_starts, fix_durations



def analyze_and_plot_neural_response_to_face_fixations_during_mutual_bouts(
        eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, 
        mutual_behav_density_df, params):
    """Runs neural response analysis for mutual face fixation periods, parallelizing across sessions if enabled."""
    logger.info("Starting neural analysis and individual unit plotting for mutual face fixation periods")
    # Set up plot save directory
    today_date = datetime.today().strftime('%Y-%m-%d')
    min_consecutive_sig_bins = params.get('min_consecutive_sig_bins', 5)
    min_total_sig_bins = params.get('min_total_sig_bins', 25)
    today_date += f'_{min_consecutive_sig_bins}-{min_total_sig_bins}_minbin'
    root_dir = os.path.join(params['root_data_dir'], "plots", "neural_response_mutual_face_fix", today_date)
    os.makedirs(root_dir, exist_ok=True)
    # Get unique session names
    session_names = eye_mvm_behav_df['session_name'].unique()
    parallelize = params.get('parallelize_over_sessions', False)
    num_processes = params.get('available_cpus', 1)
    logger.info(f"Parallelizing over sessions: {parallelize} | Num processes assigned: {num_processes}")
    if parallelize:
        results = Parallel(n_jobs=num_processes)(
            delayed(process_neural_response_to_face_fixations_for_session)(
                session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, 
                mutual_behav_density_df, root_dir, params
            ) for session_name in tqdm(session_names, desc="Processing sessions in parallel")
        )
    else:
        results = [
            process_neural_response_to_face_fixations_for_session(
                session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, 
                mutual_behav_density_df, root_dir, params
            ) for session_name in tqdm(session_names, desc="Processing sessions in serial")
        ]
    # Merge results
    merged_sig_units, merged_non_sig_units = {}, {}
    merged_high_density_counts, merged_low_density_counts = {}, {}
    for sig_units, non_sig_units, high_density_counts, low_density_counts in results:
        for region, units in sig_units.items():
            merged_sig_units.setdefault(region, []).extend(units)
        for region, units in non_sig_units.items():
            merged_non_sig_units.setdefault(region, []).extend(units)
        for region, unit_counts in high_density_counts.items():
            merged_high_density_counts.setdefault(region, []).extend(unit_counts)
        for region, unit_counts in low_density_counts.items():
            merged_low_density_counts.setdefault(region, []).extend(unit_counts)
    # Compute summary counts
    summary_counts = {
        region: (len(merged_sig_units.get(region, [])), len(merged_sig_units.get(region, [])) + len(merged_non_sig_units.get(region, [])))
        for region in set(merged_sig_units.keys()).union(set(merged_non_sig_units.keys()))
    }
    # Display results
    logger.info("Significant neuron counts by brain region:")
    for region, (sig_count, total_count) in summary_counts.items():
        logger.info(f"{region}: {sig_count} / {total_count} significant units")
    # Generate and save summary plot
    generate_summary_plot(merged_sig_units, merged_high_density_counts, merged_low_density_counts, root_dir, params)
    return merged_sig_units, merged_non_sig_units


def process_neural_response_to_face_fixations_for_session(session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, 
                                              spike_times_df, mutual_behav_density_df, root_dir, params):
    """Processes a single session, computing and plotting neural response for mutual face fixation periods."""
    logger.debug(f'Processing session {session_name}')
    min_consecutive_sig_bins = params.get('min_consecutive_sig_bins', 5)
    min_total_sig_bins = params.get('min_total_sig_bins', 25)
    sig_units, non_sig_units = {}, {}
    high_density_counts, low_density_counts = {}, {}
    session_spike_data = spike_times_df[spike_times_df['session_name'] == session_name]
    if session_spike_data.empty:
        return sig_units, non_sig_units, high_density_counts, low_density_counts
    high_density_fixations = []
    low_density_fixations = []
    # Extract session behavioral data
    session_df = eye_mvm_behav_df[eye_mvm_behav_df['session_name'] == session_name]
    for run_number in session_df['run_number'].unique():
        run_df = session_df[session_df['run_number'] == run_number]
        m1_data = run_df[run_df['agent'] == 'm1']
        if m1_data.empty:
            continue  # Skip if m1 data is missing
        # Extract fixation intervals and locations
        m1_fix_intervals = m1_data['fixation_start_stop'].iloc[0]
        m1_fix_locations = m1_data['fixation_location'].iloc[0]
        # Extract mutual density from mutual_behav_density_df
        mutual_density_data = mutual_behav_density_df[
            (mutual_behav_density_df['session_name'] == session_name) & 
            (mutual_behav_density_df['run_number'] == run_number) &
            (mutual_behav_density_df['fixation_type'] == "face")
        ]
        if mutual_density_data.empty:
            continue
        mutual_density = np.array(mutual_density_data['mutual_density'].iloc[0]).flatten()
        # Extract neural timeline from synchronized gaze data
        gaze_data = sparse_nan_removed_sync_gaze_df[
            (sparse_nan_removed_sync_gaze_df['session_name'] == session_name) & 
            (sparse_nan_removed_sync_gaze_df['run_number'] == run_number) &
            (sparse_nan_removed_sync_gaze_df['agent'] == 'm1')  
        ]
        if gaze_data.empty:
            continue
        neural_timeline = np.array(gaze_data['neural_timeline'].iloc[0]).flatten()
        # Determine threshold for high/low density fixations
        mutual_density_threshold = np.mean(mutual_density) * 0.63
        # Identify face fixations of M1 in high vs. low mutual density periods
        for (start, stop), location in zip(m1_fix_intervals, m1_fix_locations):
            if 'face' in location:
                fixation_time_in_neurons = neural_timeline[start]  # Convert to neural time
                if mutual_density[start] > mutual_density_threshold:
                    high_density_fixations.append(fixation_time_in_neurons)
                else:
                    low_density_fixations.append(fixation_time_in_neurons)
    # Process each unit's neural response
    for _, unit in session_spike_data.iterrows():
        unit_uuid = unit["unit_uuid"]
        brain_region = unit["region"]
        logger.debug(f'Processing {unit_uuid} in {brain_region}')
        spike_times = np.array(unit["spike_ts"])
        is_sig = 0
        # Compute spike counts per trial
        high_density_spike_counts, timeline = compute_spike_counts_per_fixation(
            high_density_fixations, spike_times, params
        )
        low_density_spike_counts, _ = compute_spike_counts_per_fixation(
            low_density_fixations, spike_times, params
        )
        # Store spike counts for later processing
        high_density_counts.setdefault(brain_region, []).append((unit_uuid, high_density_spike_counts))
        low_density_counts.setdefault(brain_region, []).append((unit_uuid, low_density_spike_counts))
        # Perform statistical test
        significant_bins = perform_ttest_with_fdr(
            high_density_spike_counts, low_density_spike_counts, timeline, params
        )
        # Identify significant units
        groups = np.split(significant_bins, np.where(np.diff(significant_bins) > 1)[0] + 1)
        longest_consec_sig_bin_count = max(len(g) for g in groups) if groups else 0
        if (longest_consec_sig_bin_count >= min_consecutive_sig_bins) or (len(significant_bins) >= min_total_sig_bins):
            is_sig = 1
            sig_units.setdefault(brain_region, []).append(unit_uuid)
        else:
            non_sig_units.setdefault(brain_region, []).append(unit_uuid)
        # Plot results
        plot_neural_response_to_high_and_low_density_face_fixations(
            timeline, high_density_spike_counts, low_density_spike_counts, significant_bins, 
            session_name, unit_uuid, brain_region, root_dir, is_sig
        )
    return sig_units, non_sig_units, high_density_counts, low_density_counts



def compute_spike_counts_per_fixation(fixation_times, spike_times, params, do_parallel=False):
    """Computes spike counts per fixation trial, optionally in parallel."""
    spike_counts_per_trial = []
    bin_size = params["neural_data_bin_size"]
    sigma = params["gaussian_smoothing_sigma"]
    time_window = params['time_window_before_and_after_event_for_psth']
    timeline = np.arange(-time_window, time_window + bin_size, bin_size)
    
    def get_spike_counts_for_fixation(fixation_time):
        bins = np.linspace(
            fixation_time - time_window,
            fixation_time + time_window,
            int(2 * time_window / bin_size) + 1
        ).ravel()
        spike_counts, _ = np.histogram(spike_times, bins=bins)
        spike_counts = spike_counts / bin_size  # Convert to firing rate
        if params["smooth_spike_counts"]:
            spike_counts = gaussian_filter1d(spike_counts, sigma=sigma)
        return spike_counts

    if do_parallel:
        threads_per_cpu = params.get('threads_per_cpu', 1)
        with parallel_backend("threading"):
            spike_counts_per_trial = Parallel(n_jobs=threads_per_cpu)(
                delayed(get_spike_counts_for_fixation)(fixation_time) for fixation_time in fixation_times
            )
    else:
        spike_counts_per_trial = [get_spike_counts_for_fixation(fixation_time) for fixation_time in fixation_times]
    return np.array(spike_counts_per_trial), timeline



def perform_ttest_with_fdr(high_density_spike_counts, low_density_spike_counts, timeline, params, fdr_correct=False, do_parallel=False):
    """Performs independent t-tests per bin, optionally in parallel, and applies FDR correction."""
    num_bins = len(timeline) - 1
    p_values = [1.0] * num_bins  # Default non-significant p-values
    time_window_start, time_window_end = -0.45, 0.45
    valid_bins = (timeline[:-1] >= time_window_start) & (timeline[:-1] <= time_window_end)
    
    def do_ttest_for_time_bin(bin_idx):
        try:
            return ttest_ind(
                high_density_spike_counts[:, bin_idx],
                low_density_spike_counts[:, bin_idx],
                equal_var=False  # Welch's t-test (recommended if variance is unequal)
            )[1]  # Extract p-value
        except ValueError:
            return 1.0  # If no valid samples exist, return non-significant p-value

    if do_parallel:
        threads_per_cpu = params.get('threads_per_cpu', 1)
        with parallel_backend("threading"):
            p_values = Parallel(n_jobs=threads_per_cpu)(
                delayed(do_ttest_for_time_bin)(bin_idx) for bin_idx in range(num_bins)
            )
    else:
        p_values = [do_ttest_for_time_bin(bin_idx) for bin_idx in range(num_bins)]
    # Apply FDR correction
    corrected_p_values = multipletests(p_values, alpha=0.05, method="fdr_bh")[1] if fdr_correct else np.array(p_values)
    significant_bins = np.where((corrected_p_values < 0.05) & valid_bins)[0]
    return significant_bins



def perform_wilcoxon_with_fdr(high_density_spike_counts, low_density_spike_counts, timeline, params, fdr_correct=False, do_parallel=False):
    """Performs Wilcoxon signed-rank test per bin, optionally in parallel, and applies FDR correction."""
    num_bins = len(timeline) - 1
    p_values = [1.0] * num_bins  # Default non-significant p-values
    time_window_start, time_window_end = -0.45, 0.45
    valid_bins = (timeline[:-1] >= time_window_start) & (timeline[:-1] <= time_window_end)
    
    def do_wicoxon_for_time_bin(bin_idx):
        try:
            return wilcoxon(
                high_density_spike_counts[:, bin_idx],
                low_density_spike_counts[:, bin_idx]
            )[1]
        except ValueError:
            return 1.0  # If no valid pairs exist, return non-significant p-value

    if do_parallel:
        threads_per_cpu = params.get('threads_per_cpu', 1)
        with parallel_backend("threading"):
            p_values = Parallel(n_jobs=threads_per_cpu)(
                delayed(do_wicoxon_for_time_bin)(bin_idx) for bin_idx in range(num_bins)
            )
    else:
        p_values = [do_wicoxon_for_time_bin(bin_idx) for bin_idx in range(num_bins)]
    # Apply FDR correction
    corrected_p_values = multipletests(p_values, alpha=0.05, method="fdr_bh")[1] if fdr_correct else np.array(p_values)
    significant_bins = np.where((corrected_p_values < 0.05) & valid_bins)[0]
    return significant_bins



def plot_neural_response_to_high_and_low_density_face_fixations(
    timeline, high_density_spike_counts, low_density_spike_counts, significant_bins, 
    session_name, unit_uuid, brain_region, root_dir, is_sig):
    """Plots the neural response to face fixations in high vs. low mutual density periods."""
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



def generate_summary_plot(merged_sig_units, merged_high_density_counts, merged_low_density_counts, root_dir, params):
    """Generates a summary plot of the timecourse index for all significant units in each region."""
    logger.info("Generating summary plot for significant neurons")
    fig, axes = plt.subplots(len(merged_sig_units), 1, figsize=(10, 5 * len(merged_sig_units)), constrained_layout=True)
    if len(merged_sig_units) == 1:
        axes = [axes]
    for ax, (region, units) in zip(axes, merged_sig_units.items()):
        unit_indices = []
        timecourses = []
        for unit_uuid, high_density_spike_counts in merged_high_density_counts.get(region, []):
            _, low_density_spike_counts = next(
                (u, c) for u, c in merged_low_density_counts.get(region, []) if u == unit_uuid
            )
            timeline = np.arange(len(high_density_spike_counts[0]))
            index = (np.mean(high_density_spike_counts, axis=0) - np.mean(low_density_spike_counts, axis=0)) / \
                    (np.mean(high_density_spike_counts, axis=0) + np.mean(low_density_spike_counts, axis=0) + 1e-6)
            max_val = np.max(np.abs(index))
            max_loc = timeline[np.argmax(np.abs(index))]
            unit_indices.append((unit_uuid, max_val, max_loc, index))
            timecourses.append(index)
        if not timecourses:
            continue
        # Sort units by magnitude and location of max index value
        unit_indices.sort(key=lambda x: (-x[1], x[2]))
        sorted_indices = [idx[-1] for idx in unit_indices]
        # Create color-coded heatmap
        im = ax.imshow(sorted_indices, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='none')
        ax.set_title(f"{region}: Significant Units")
        ax.set_xlabel("Time from Fixation (ms)")
        ax.set_ylabel("Unit (sorted by peak response)")
        fig.colorbar(im, ax=ax, label="Normalized Index")
    summary_plot_path = os.path.join(root_dir, "summary_neural_response.png")
    plt.savefig(summary_plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Summary plot saved at {summary_plot_path}")




























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