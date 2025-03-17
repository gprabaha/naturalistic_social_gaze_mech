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
import pickle

import seaborn as sns
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        'reload_processed_data': True,
        'parallelize_over_sessions': False,
        'recalculate_mutual_density_df': False,
        'fixation_type_to_process': 'face',
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



def analyze_and_plot_neural_response_to_face_and_object_fixations(
        eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, 
        mutual_behav_density_df, params):
    """Runs neural response analysis for mutual face and object fixation periods, parallelizing across sessions if enabled."""
    logger.info("Starting neural analysis and individual unit plotting for mutual face and object fixation periods")
    
    # Set up directories
    today_date = datetime.today().strftime('%Y-%m-%d')
    min_consecutive_sig_bins = params.get('min_consecutive_sig_bins', 5)
    min_total_sig_bins = params.get('min_total_sig_bins', 25)
    today_date += f'_{min_consecutive_sig_bins}-{min_total_sig_bins}_minbin'
    root_dir = os.path.join(params['root_data_dir'], "plots", "neural_response_face_object_fix", today_date)
    os.makedirs(root_dir, exist_ok=True)
    processed_data_dir = params['processed_data_dir']
    processed_data_path = os.path.join(processed_data_dir, "neural_response_data.pkl")
    
    # Check if we should reload existing processed data
    if params.get('reload_processed_data', False) and os.path.exists(processed_data_path):
        logger.info("Loading precomputed neural response data from disk.")
        with open(processed_data_path, 'rb') as f:
            merged_sig_units, merged_non_sig_units, merged_high_density_counts, merged_low_density_counts, merged_object_counts, merged_face_vs_object_sig_units, timeline = pickle.load(f)
    else:
        # Get unique session names
        session_names = eye_mvm_behav_df['session_name'].unique()
        parallelize = params.get('parallelize_over_sessions', False)
        num_processes = params.get('available_cpus', 1)
        logger.info(f"Parallelizing over sessions: {parallelize} | Num processes assigned: {num_processes}")
        
        if parallelize:
            results = Parallel(n_jobs=num_processes)(
                delayed(process_neural_response_to_face_and_object_fixations_for_session)(
                    session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, 
                    mutual_behav_density_df, root_dir, params
                ) for session_name in tqdm(session_names, desc="Processing sessions in parallel")
            )
        else:
            results = [
                process_neural_response_to_face_and_object_fixations_for_session(
                    session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, 
                    mutual_behav_density_df, root_dir, params
                ) for session_name in tqdm(session_names, desc="Processing sessions in serial")
            ]
        
        # Merge results
        merged_sig_units, merged_non_sig_units = {}, {}
        merged_high_density_counts, merged_low_density_counts = {}, {}
        merged_object_counts, merged_face_vs_object_sig_units = {}, {}
        
        for sig_units, non_sig_units, high_density_counts, low_density_counts, object_counts, face_vs_object_sig_units, timeline in results:
            for region, units in sig_units.items():
                merged_sig_units.setdefault(region, []).extend(units)
            for region, units in non_sig_units.items():
                merged_non_sig_units.setdefault(region, []).extend(units)
            for region, unit_counts in high_density_counts.items():
                merged_high_density_counts.setdefault(region, []).extend(unit_counts)
            for region, unit_counts in low_density_counts.items():
                merged_low_density_counts.setdefault(region, []).extend(unit_counts)
            for region, unit_counts in object_counts.items():
                merged_object_counts.setdefault(region, []).extend(unit_counts)
            for region, units in face_vs_object_sig_units.items():
                merged_face_vs_object_sig_units.setdefault(region, []).extend(units)
        
        # Save processed data
        logger.info("Saving processed neural response data to disk.")
        with open(processed_data_path, 'wb') as f:
            pickle.dump((merged_sig_units, merged_non_sig_units, merged_high_density_counts, merged_low_density_counts, merged_object_counts, merged_face_vs_object_sig_units, timeline), f)
    
    # Compute summary counts for high vs low face fixations
    summary_counts = {
        region: (len(merged_sig_units.get(region, [])), len(merged_sig_units.get(region, [])) + len(merged_non_sig_units.get(region, [])))
        for region in set(merged_sig_units.keys()).union(set(merged_non_sig_units.keys()))
    }
    
    # Compute summary counts for face vs object fixations
    summary_counts_face_vs_object = {
        region: len(merged_face_vs_object_sig_units.get(region, []))
        for region in merged_face_vs_object_sig_units.keys()
    }
    
    # Display results
    logger.info("Significant neuron counts by brain region (high vs low density face fixations):")
    for region, (sig_count, total_count) in summary_counts.items():
        percentage = (sig_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"{region}: {sig_count} / {total_count} significant units ({percentage:.2f}%)")
    
    logger.info("Significant neuron counts by brain region (face vs object fixations):")
    for region, sig_count in summary_counts_face_vs_object.items():
        logger.info(f"{region}: {sig_count} significant units")
    
    # Compute and display percentage overlap
    logger.info("Overlap between face vs object and high vs low density classifiers:")
    for region in merged_sig_units.keys():
        face_vs_object_set = set(merged_face_vs_object_sig_units.get(region, []))
        high_vs_low_set = set(merged_sig_units.get(region, []))
        overlap_count = len(face_vs_object_set & high_vs_low_set)
        total_count = len(face_vs_object_set | high_vs_low_set)
        overlap_percentage = (overlap_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"{region}: {overlap_count} overlapping units ({overlap_percentage:.2f}% overlap)")
    
    # Generate and save summary plots
    generate_summary_plots(merged_sig_units, merged_high_density_counts, merged_low_density_counts, merged_object_counts, timeline, root_dir)
    plot_pca_trajectory(merged_high_density_counts, merged_low_density_counts, merged_object_counts, timeline, root_dir)
    
    return merged_sig_units, merged_non_sig_units, merged_face_vs_object_sig_units




def process_neural_response_to_face_and_object_fixations_for_session(session_name, eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, 
                                                                    spike_times_df, mutual_behav_density_df,
                                                                    root_dir, params):
    """Processes a single session, computing and plotting neural response for mutual face and object fixation periods."""
    logger.debug(f'Processing session {session_name}')
    min_consecutive_sig_bins = params.get('min_consecutive_sig_bins', 5)
    min_total_sig_bins = params.get('min_total_sig_bins', 25)
    sig_units, non_sig_units = {}, {}
    high_density_counts, low_density_counts = {}, {}
    object_counts = {}
    face_vs_object_sig_units = {}
    session_spike_data = spike_times_df[spike_times_df['session_name'] == session_name]
    if session_spike_data.empty:
        return sig_units, non_sig_units, high_density_counts, low_density_counts, object_counts, face_vs_object_sig_units
    high_density_fixations, low_density_fixations, object_fixations = [], [], []
    
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
        
        # Identify face and object fixations
        for (start, stop), location in zip(m1_fix_intervals, m1_fix_locations):
            fixation_time_in_neurons = neural_timeline[start]  # Convert to neural time
            if 'face' in location:
                if mutual_density[start] > mutual_density_threshold:
                    high_density_fixations.append(fixation_time_in_neurons)
                else:
                    low_density_fixations.append(fixation_time_in_neurons)
            elif 'object' in location:
                object_fixations.append(fixation_time_in_neurons)
    
    # Process each unit's neural response
    for _, unit in session_spike_data.iterrows():
        unit_uuid = unit["unit_uuid"]
        brain_region = unit["region"]
        logger.debug(f'Processing {unit_uuid} in {brain_region}')
        spike_times = np.array(unit["spike_ts"])
        
        # Compute spike counts per trial
        high_density_spike_counts, timeline = compute_spike_counts_per_fixation(
            high_density_fixations, spike_times, params
        )
        low_density_spike_counts, _ = compute_spike_counts_per_fixation(
            low_density_fixations, spike_times, params
        )
        object_spike_counts, _ = compute_spike_counts_per_fixation(
            object_fixations, spike_times, params
        )
        
        # Store spike counts for later processing
        high_density_counts.setdefault(brain_region, []).append((unit_uuid, high_density_spike_counts))
        low_density_counts.setdefault(brain_region, []).append((unit_uuid, low_density_spike_counts))
        object_counts.setdefault(brain_region, []).append((unit_uuid, object_spike_counts))
        
        # Perform statistical tests
        significant_bins_face_density = perform_ttest_with_fdr(
            high_density_spike_counts, low_density_spike_counts, timeline, params
        )
        significant_bins_face_vs_object = perform_ttest_with_fdr(
            high_density_spike_counts + low_density_spike_counts, object_spike_counts, timeline, params
        )
        
        # Identify significant units
        def classify_units_by_significance(significant_bins, unit_uuid, brain_region, min_consecutive, min_total, sig_units, non_sig_units):
            groups = np.split(significant_bins, np.where(np.diff(significant_bins) > 1)[0] + 1)
            longest_consec_sig_bin_count = max(len(g) for g in groups) if groups else 0
            if (longest_consec_sig_bin_count >= min_consecutive) or (len(significant_bins) >= min_total):
                sig_units.setdefault(brain_region, []).append(unit_uuid)
            else:
                non_sig_units.setdefault(brain_region, []).append(unit_uuid)
            return sig_units, non_sig_units
        
        sig_units, non_sig_units = classify_units_by_significance(
            significant_bins_face_density, unit_uuid, brain_region, min_consecutive_sig_bins, min_total_sig_bins, sig_units, non_sig_units
        )
        face_vs_object_sig_units = classify_units_by_significance(
            significant_bins_face_vs_object, unit_uuid, brain_region, min_consecutive_sig_bins, min_total_sig_bins, face_vs_object_sig_units, {}
        )
        
        # # Plot results
        # plot_neural_response_to_high_and_low_density_face_fixations(
        #     timeline, high_density_spike_counts, low_density_spike_counts, significant_bins_face_density, 
        #     session_name, unit_uuid, brain_region, root_dir, int(unit_uuid in sig_units.get(brain_region, []))
        # )
        # plot_neural_response_to_face_vs_object_fixations(
        #     timeline, high_density_spike_counts + low_density_spike_counts, object_spike_counts, significant_bins_face_vs_object, 
        #     session_name, unit_uuid, brain_region, root_dir, int(unit_uuid in face_vs_object_sig_units.get(brain_region, []))
        # )
    
    return sig_units, non_sig_units, high_density_counts, low_density_counts, object_counts, face_vs_object_sig_units, timeline




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



def generate_summary_plots(merged_sig_units, merged_high_density_counts, merged_low_density_counts, timeline, root_dir):
    """Generates multiple summary plots of the timecourse index for all significant units in each region using different sorting methods."""
    logger.info("Generating summary plots for significant neurons with different sorting methods")

    # Define sorting methods
    sort_methods = {
        "bias": lambda x: (-x[2],),
        "max_loc": lambda x: (x[3],),
        "min_loc": lambda x: (x[5],),
        "abs_max_loc": lambda x: (min(x[3], x[5], key=abs),)
    }

    # Precompute indices for each region
    precomputed_indices = {}

    for region, units in merged_sig_units.items():
        unit_indices = []
        for unit_uuid, high_density_spike_counts in merged_high_density_counts.get(region, []):
            if unit_uuid in units:
                try:
                    _, low_density_spike_counts = next(
                        (u, c) for u, c in merged_low_density_counts.get(region, []) if u == unit_uuid
                    )
                except StopIteration:
                    continue  # Skip units without corresponding low-density data

                # Compute normalized index
                index = (np.mean(high_density_spike_counts, axis=0) - np.mean(low_density_spike_counts, axis=0)) / \
                        (np.mean(high_density_spike_counts, axis=0) + np.mean(low_density_spike_counts, axis=0) + 1e-6)

                # Extract key properties for sorting
                max_val, max_idx = np.max(index), np.argmax(index)
                min_val, min_idx = np.min(index), np.argmin(index)
                max_loc, min_loc = timeline[max_idx], timeline[min_idx]
                bias = max_val - min_val  # Measure of bias towards positive
                
                unit_indices.append((unit_uuid, max_val, bias, max_loc, min_val, min_loc, index))

        if unit_indices:
            precomputed_indices[region] = unit_indices

    # Now iterate over different sorting methods
    for sort_name, sort_key in sort_methods.items():
        num_regions = len(precomputed_indices)
        num_rows, num_cols = 2, 2  # Fixed 2x2 grid
        total_plots = min(num_rows * num_cols, num_regions)

        fig = plt.figure(figsize=(20, 16))  # Larger figure for spacing
        gs = gridspec.GridSpec(num_rows * 2, num_cols * 2, figure=fig, width_ratios=[1, 0.2] * num_cols, height_ratios=[0.2, 1] * num_rows)

        # Set global title for sorting method
        fig.suptitle(
            f"Neural Response Index Summary\n"
            f"(Sorted by {sort_name.replace('_', ' ').title()})\n"
            r"Index = (High-Density - Low-Density) / (High-Density + Low-Density)",
            fontsize=16, fontweight='bold', ha='center'
        )


        axes = []  # To store main axes
        for i in range(num_rows):
            for j in range(num_cols):
                axes.append(fig.add_subplot(gs[i * 2 + 1, j * 2]))  # Main heatmaps

        for ax in axes:  
            ax.axis('off')  # Hide unused subplots

        for idx, (ax, (region, unit_indices)) in enumerate(zip(axes[:total_plots], precomputed_indices.items())):
            # Sort based on the selected method
            unit_indices_sorted = sorted(unit_indices, key=sort_key)
            sorted_indices = np.array([idx[-1] for idx in unit_indices_sorted])  # Extract sorted data
            
            # If sorting by abs max location, take absolute values
            if sort_name == "abs_max_loc":
                sorted_indices = np.abs(sorted_indices)

            # Compute marginals
            top_marginal = np.mean(sorted_indices, axis=0)  # Mean across rows (time-wise average)
            right_marginal = np.mean(sorted_indices, axis=1)  # Mean across columns (unit-wise average)

            # If sorting by abs max location, ensure marginals also use absolute values
            if sort_name == "abs_max_loc":
                top_marginal = np.abs(top_marginal)
                right_marginal = np.abs(right_marginal)

            # Choose colormap
            if sort_name == "abs_max_loc":
                cmap = 'Reds'  # Use only red instead of red-blue
                vmin, vmax = 0, 1
            else:
                cmap = 'RdBu_r'  # Red-Blue colormap
                vmin, vmax = -1, 1

            # Create color-coded heatmap
            im = ax.imshow(sorted_indices, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none',
                           extent=[timeline[0], timeline[-1], 0, len(sorted_indices)])  # Aligns x-axis to timeline

            # Draw vertical line at t=0
            ax.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.7)

            # X-axis time labels
            ax.set_xticks(np.linspace(timeline[0], timeline[-1], num=6))  # 6 evenly spaced ticks
            ax.set_xticklabels(np.round(np.linspace(timeline[0], timeline[-1], num=6)).astype(int))
            ax.set_xlabel("Time from Fixation (ms)")

            # Marginal plot: Top (Vertical bars, now correctly aligned)
            top_ax = fig.add_subplot(gs[idx // num_cols * 2, idx % num_cols * 2])  # Top marginal subplot
            top_ax.bar(timeline[:-1], top_marginal, width=(timeline[1] - timeline[0]), color='black', alpha=0.8, align='edge')  # Thin black bars
            top_ax.set_xlim([timeline[0], timeline[-1]])
            top_ax.set_ylim([0, np.max(top_marginal) * 1.1])  # Adjusted limit to prevent overlap
            top_ax.set_xticks([])
            top_ax.set_yticks([])
            top_ax.axis("off")

            # Marginal plot: Right (Horizontal bars, now correctly aligned)
            right_ax = fig.add_subplot(gs[idx // num_cols * 2 + 1, idx % num_cols * 2 + 1])  # Right marginal subplot
            right_ax.barh(np.arange(len(sorted_indices)), right_marginal, height=1, color='black', alpha=0.8, align='edge')  # Thin black bars
            right_ax.set_xlim([0, np.max(right_marginal) * 1.1])  # Adjusted limit to prevent overlap
            right_ax.set_ylim([0, len(sorted_indices)])
            right_ax.set_xticks([])
            right_ax.set_yticks([])
            right_ax.axis("off")

            # Titles and labels
            ax.set_title(f"{region}: {len(unit_indices_sorted)} Sig. Units")
            ax.set_ylabel("Unit Index")
            fig.colorbar(im, ax=ax, label="Normalized Index")

        # Save plot with the sorting method in the filename (High Resolution 600 dpi)
        summary_plot_path = os.path.join(root_dir, f"0_summary_neural_response_{sort_name}.png")
        plt.savefig(summary_plot_path, dpi=600, bbox_inches='tight')  # Higher DPI for clarity
        plt.close(fig)
        logger.info(f"Summary plot saved at {summary_plot_path}")



def plot_pca_trajectory(merged_high_density_counts, merged_low_density_counts, timeline, root_dir):
    """
    Computes PCA on the concatenated high- and low-density mean spike counts across all units
    for each region separately and plots the trajectory in the top three PCs.
    Also computes PCA on the (high-low)/(high+low) timeseries and plots separately.
    """
    logger.info("Performing PCA on high- and low-density neural activity trajectories")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'projection': '3d'})
    fig.suptitle("Neural Response to Face Fixations by M1 During High vs Low Interactiveness", fontsize=16, fontweight='bold')
    
    fig_index, index_axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'projection': '3d'})
    fig_index.suptitle("PCA of Normalized Neural Response Index: (High - Low) / (High + Low)", fontsize=16, fontweight='bold')
    
    # Define colormaps
    high_density_cmap = cm.viridis
    low_density_cmap = cm.magma
    index_cmap = cm.plasma
    
    regions = list(merged_high_density_counts.keys())
    
    for idx, (region, ax, index_ax) in enumerate(zip(regions, axes.flatten(), index_axes.flatten())):
        all_high_density = []
        all_low_density = []
        all_index = []
        
        for unit_uuid, high_density_spike_counts in merged_high_density_counts.get(region, []):
            low_density_spike_counts = next((c for u, c in merged_low_density_counts.get(region, []) if u == unit_uuid), None)
            if low_density_spike_counts is None:
                continue  # Skip units without corresponding low-density data

            mean_high = np.mean(high_density_spike_counts, axis=0)
            mean_low = np.mean(low_density_spike_counts, axis=0)
            index_series = (mean_high - mean_low) / (mean_high + mean_low + 1e-6)
            
            all_high_density.append(mean_high)
            all_low_density.append(mean_low)
            all_index.append(index_series)
        
        if not all_high_density or not all_low_density:
            logger.warning(f"No valid units found for region {region}.")
            continue
        
        all_high_density = np.array(all_high_density)
        all_low_density = np.array(all_low_density)
        all_index = np.array(all_index)
        
        # PCA for High vs Low Density
        data_matrix = np.hstack([all_high_density, all_low_density])
        pca = PCA(n_components=3)
        projected_data = pca.fit_transform(data_matrix.T)
        num_timepoints = len(timeline)
        projected_high = projected_data[:num_timepoints]
        projected_low = projected_data[num_timepoints:]
        
        for i in range(num_timepoints - 1):
            ax.plot(projected_high[i:i+2, 0], projected_high[i:i+2, 1], projected_high[i:i+2, 2],
                    color=high_density_cmap(0.3 + 0.7 * (i / num_timepoints)), alpha=0.8, linewidth=2,
                    label="High-density" if i == 0 else "")
            ax.plot(projected_low[i:i+2, 0], projected_low[i:i+2, 1], projected_low[i:i+2, 2],
                    color=low_density_cmap(0.3 + 0.7 * (i / num_timepoints)), alpha=0.8, linewidth=2,
                    label="Low-density" if i == 0 else "")
        
        ax.scatter(*projected_high[0], color='black', marker='o', s=80, label='Start')
        ax.scatter(*projected_high[num_timepoints // 2], color='black', marker='*', s=120, label='Fixation Onset')
        ax.scatter(*projected_high[-1], color='black', marker='s', s=80, label='End')
        
        ax.scatter(*projected_low[0], color='black', marker='o', s=80)
        ax.scatter(*projected_low[num_timepoints // 2], color='black', marker='*', s=120)
        ax.scatter(*projected_low[-1], color='black', marker='s', s=80)
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{region} - High vs Low Density")
        ax.legend()
        
        # PCA for Normalized Index
        pca_index = PCA(n_components=3)
        projected_index = pca_index.fit_transform(all_index.T)
        
        for i in range(num_timepoints - 1):
            index_ax.plot(projected_index[i:i+2, 0], projected_index[i:i+2, 1], projected_index[i:i+2, 2],
                          color=index_cmap(0.3 + 0.7 * (i / num_timepoints)), alpha=0.8, linewidth=2)
        
        index_ax.scatter(*projected_index[0], color='black', marker='o', s=80, label='Start')
        index_ax.scatter(*projected_index[num_timepoints // 2], color='black', marker='*', s=120, label='Fixation Onset')
        index_ax.scatter(*projected_index[-1], color='black', marker='s', s=80, label='End')
        
        index_ax.set_xlabel("PC1")
        index_ax.set_ylabel("PC2")
        index_ax.set_zlabel("PC3")
        index_ax.set_title(f"{region} - Normalized Index")
        index_ax.legend()
    
    pca_plot_path = os.path.join(root_dir, "0_pca_trajectory_high_low_density.png")
    fig.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"PCA trajectory plot saved at {pca_plot_path}")
    
    pca_index_plot_path = os.path.join(root_dir, "0_pca_index_trajectory.png")
    fig_index.savefig(pca_index_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig_index)
    logger.info(f"PCA index trajectory plot saved at {pca_index_plot_path}")




if __name__ == "__main__":
    main()