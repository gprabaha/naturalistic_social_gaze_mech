import numpy as np
import pandas as pd
import os
import logging
import psutil
import subprocess
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import sem
from scipy.stats import ttest_1samp
from scipy.stats import wilcoxon

import pdb

import load_data
import curate_data


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)



# ** Initiation and Main **



def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'is_cluster': True,
        'is_grace': False,
        'recompute_fix_binary_vector': False,
        'recompute_crosscorr': False,
        'remake_plots': True
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    logger.info("Parameters initialized successfully")
    return params



def main():
    logger.info("Starting the script")
    params = _initialize_params()

    processed_data_dir = params['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)  # Ensure the directory exists

    eye_mvm_behav_df_file_path = os.path.join(processed_data_dir, 'eye_mvm_behav_df.pkl')
    monkeys_per_session_dict_file_path = os.path.join(processed_data_dir, 'ephys_days_and_monkeys.pkl')

    logger.info("Loading data files")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    monkeys_per_session_df = pd.DataFrame(load_data.get_data_df(monkeys_per_session_dict_file_path))
    logger.info("Data loaded successfully")

    num_cpus, threads_per_cpu = get_slurm_cpus_and_threads()
    
    # Fix-related binary vectors
    fix_binary_vector_file = os.path.join(processed_data_dir, 'fix_binary_vector_df.pkl')
        
    if params.get('recompute_fix_binary_vector', False):
        logger.info("Generating fix-related binary vectors")
        fix_binary_vector_df = generate_fixation_binary_vectors(eye_mvm_behav_df)
        fix_binary_vector_df.to_pickle(fix_binary_vector_file)
        logger.info(f"Fix-related binary vectors computed and saved to {fix_binary_vector_file}")
    else:
        logger.info("Loading precomputed fix-related binary vectors")
        fix_binary_vector_df = load_data.get_data_df(fix_binary_vector_file)

    # Cross-correlation computation
    inter_agent_cross_corr_file = os.path.join(processed_data_dir, 'inter_agent_behav_cross_correlation_df.pkl')

    if params.get('recompute_crosscorr', False):
        logger.info("Computing cross-correlations and shuffled statistics")
        inter_agent_behav_cross_correlation_df = compute_regular_and_shuffled_crosscorr_parallel(
            fix_binary_vector_df, eye_mvm_behav_df, sigma=3, num_shuffles=5000, num_cpus=num_cpus, threads_per_cpu=threads_per_cpu
        )
        inter_agent_behav_cross_correlation_df.to_pickle(inter_agent_cross_corr_file)
        logger.info(f"Cross-correlations and shuffled statistics computed and saved to {inter_agent_cross_corr_file}")
    else:
        logger.info("Loading precomputed cross-correlations and shuffled statistics")
        inter_agent_behav_cross_correlation_df = load_data.get_data_df(inter_agent_cross_corr_file)

    if params.get('remake_plots', False):

        # logger.info("Plotting cross-correlation timecourse averaged across sessions")
        # plot_fixation_crosscorr_minus_shuffled(inter_agent_behav_cross_correlation_df, monkeys_per_session_df, params, 
        #                     group_by="session_name", plot_duration_seconds=90)
        # logger.info("Plotting cross-correlation timecourse averaged across monkey pairs")
        # plot_fixation_crosscorr_minus_shuffled(inter_agent_behav_cross_correlation_df, monkeys_per_session_df, params, 
        #                     group_by="monkey_pair", plot_duration_seconds=90)

        # logger.info("Plotting significant cross-correlation timecourse averaged across sessions")
        # plot_significant_fixation_crosscorr_minus_shuffled(
        #     inter_agent_behav_cross_correlation_df, monkeys_per_session_df,
        #     params, group_by="session_name", plot_duration_seconds=90, alpha=0.05)
        logger.info("Plotting significant cross-correlation timecourse averaged across monkey-pairs")
        plot_significant_fixation_crosscorr_minus_shuffled(
            inter_agent_behav_cross_correlation_df, monkeys_per_session_df,
            params, group_by="monkey_pair", plot_duration_seconds=90, alpha=0.05)
        
        logger.info("Finished cross-correlation timecourse plot generation")



# ** Sub-functions **

## Count available CPUs and threads
def get_slurm_cpus_and_threads():
    """Returns the number of allocated CPUs and threads per CPU using psutil."""
    
    # Get number of CPUs allocated by SLURM
    num_cpus = 1  # Default if not in a SLURM job
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        result = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True)
        match = re.search(r"NumCPUs=(\d+)", result.stdout)
        if match:
            num_cpus = int(match.group(1))

    # Get total virtual CPUs (logical cores)
    total_logical_cpus = psutil.cpu_count(logical=True)

    # Calculate threads per CPU
    threads_per_cpu = total_logical_cpus // num_cpus if num_cpus > 0 else 1

    return num_cpus, threads_per_cpu


## Fixation timeline binary vector generation
def generate_fixation_binary_vectors(df):
    """
    Generate a long-format dataframe with binary vectors for each fixation category.

    Args:
        df (pd.DataFrame): DataFrame with fixation start-stop indices and fixation locations.

    Returns:
        pd.DataFrame: A long-format DataFrame where each row represents a fixation type's binary vector.
    """
    binary_vectors = []

    for _, row in tqdm(df.iterrows(), desc="Processing df row"):
        run_length = row["run_length"]
        fixation_start_stop = row["fixation_start_stop"]
        fixation_categories = categorize_fixations(row["fixation_location"])

        binary_dict = {
            "eyes": np.zeros(run_length, dtype=int),
            "face": np.zeros(run_length, dtype=int),
            "out_of_roi": np.zeros(run_length, dtype=int),
        }

        for (start, stop), category in zip(fixation_start_stop, fixation_categories):
            if category == "eyes":
                binary_dict["eyes"][start:stop + 1] = 1
                binary_dict["face"][start:stop + 1] = 1  # 'face' includes 'eyes'
            elif category == "non_eye_face":
                binary_dict["face"][start:stop + 1] = 1
            elif category == "out_of_roi":
                binary_dict["out_of_roi"][start:stop + 1] = 1

        for fixation_type, binary_vector in binary_dict.items():
            binary_vectors.append({
                "session_name": row["session_name"],
                "interaction_type": row["interaction_type"],
                "run_number": row["run_number"],
                "agent": row["agent"],
                "fixation_type": fixation_type,
                "binary_vector": binary_vector
            })

    return pd.DataFrame(binary_vectors)

def categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]


## Compute crosscorrelation between m1 and m2 fixation events
def compute_regular_and_shuffled_crosscorr_parallel(fix_binary_vector_df, eye_mvm_behav_df, sigma=3, num_shuffles=100, num_cpus=16, threads_per_cpu=8):
    """
    Optimized parallelized computation of Gaussian-smoothed cross-correlation
    between m1 and m2's fixation behaviors using joblib for multiprocessing.
    """

    grouped = list(fix_binary_vector_df.groupby(["session_name", "interaction_type", "run_number", "fixation_type"]))

    # Run parallel processing across session groups
    results = Parallel(n_jobs=num_cpus)(
        delayed(compute_crosscorr_for_group)(
            group_tuple, eye_mvm_behav_df, sigma, num_shuffles, threads_per_cpu
        ) for group_tuple in tqdm(grouped, desc="Processing groups")
    )

    return pd.DataFrame([res for res in results if res])  # Filter out empty results

def compute_crosscorr_for_group(group_tuple, eye_mvm_behav_df, sigma, num_shuffles, num_threads):
    """
    Process a single (session, interaction, run, fixation_type) group.
    Uses joblib to parallelize shuffled vector generation and cross-correlation computations.
    """

    (session, interaction, run, fixation_type), group = group_tuple

    if len(group) != 2:
        return None  # Skip if both m1 and m2 aren't present
    
    m1_vector = np.array(group[group["agent"] == "m1"].iloc[0]["binary_vector"])
    m2_vector = np.array(group[group["agent"] == "m2"].iloc[0]["binary_vector"])

    # Smooth fixation vectors
    m1_smooth, m2_smooth = gaussian_filter1d(m1_vector, sigma), gaussian_filter1d(m2_vector, sigma)

    # Compute original cross-correlations
    crosscorr_m1_m2, crosscorr_m2_m1 = fft_crosscorrelation_both(m1_smooth, m2_smooth)

    # Generate shuffled vectors
    m1_shuffled_vectors = generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, "m1", num_shuffles)
    m2_shuffled_vectors = generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, "m2", num_shuffles)

    # Compute shuffled cross-correlations in parallel
    shuffled_crosscorrs = Parallel(n_jobs=num_threads)(
        delayed(fft_crosscorrelation_both)(gaussian_filter1d(m1, sigma), gaussian_filter1d(m2, sigma))
        for m1, m2 in zip(m1_shuffled_vectors, m2_shuffled_vectors)
    )

    # Convert shuffled cross-correlations into mean/std arrays
    shuffled_crosscorrs_m1_m2 = np.array([s[0] for s in shuffled_crosscorrs])
    shuffled_crosscorrs_m2_m1 = np.array([s[1] for s in shuffled_crosscorrs])

    return {
        "session_name": session,
        "interaction_type": interaction,
        "run_number": run,
        "fixation_type": fixation_type,
        "crosscorr_m1_m2": crosscorr_m1_m2,
        "crosscorr_m2_m1": crosscorr_m2_m1,
        "mean_shuffled_m1_m2": np.mean(shuffled_crosscorrs_m1_m2, axis=0),
        "std_shuffled_m1_m2": np.std(shuffled_crosscorrs_m1_m2, axis=0),
        "mean_shuffled_m2_m1": np.mean(shuffled_crosscorrs_m2_m1, axis=0),
        "std_shuffled_m2_m1": np.std(shuffled_crosscorrs_m2_m1, axis=0)
    }

def fft_crosscorrelation_both(x, y):
    """Compute cross-correlation using fftconvolve."""
    full_corr = fftconvolve(x, y[::-1], mode='full')

    n = len(x)
    mid = len(full_corr) // 2

    return full_corr[mid:mid + n], full_corr[:mid][::-1]

def generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, agent, num_shuffles):
    """
    Generate shuffled versions of a binary fixation vector by randomly redistributing fixation intervals.
    Uses joblib for parallelization.
    """

    row = eye_mvm_behav_df[
        (eye_mvm_behav_df["session_name"] == session) &
        (eye_mvm_behav_df["interaction_type"] == interaction) &
        (eye_mvm_behav_df["run_number"] == run) &
        (eye_mvm_behav_df["agent"] == agent)
    ].iloc[0]

    run_length = row["run_length"]
    categories = categorize_fixations(row["fixation_location"])
    fixation_start_stop = row["fixation_start_stop"]

    valid_categories = {"eyes", "non_eye_face"} if fixation_type == "face" else {fixation_type}

    fixation_intervals = [
        (start, stop) for (start, stop), category in zip(fixation_start_stop, categories) if category in valid_categories
    ]
    fixation_durations = [stop - start + 1 for start, stop in fixation_intervals]

    total_fixation_duration = sum(fixation_durations)
    available_non_fixation_duration = run_length - total_fixation_duration

    # Parallel execution using joblib
    shuffled_vectors = Parallel(n_jobs=num_shuffles)(
        delayed(generate_single_shuffled_vector)(
            fixation_durations, available_non_fixation_duration, len(fixation_durations), run_length
        ) for _ in range(num_shuffles)
    )

    return shuffled_vectors

def generate_single_shuffled_vector(fixation_durations, available_non_fixation_duration, N, run_length):
    """
    Generates a single shuffled binary vector by redistributing fixation intervals.
    """
    non_fixation_durations = generate_uniformly_distributed_partitions(available_non_fixation_duration, N + 1)

    all_segments = create_labeled_segments(fixation_durations, non_fixation_durations)

    np.random.shuffle(all_segments)
    shuffled_vector = construct_shuffled_vector(all_segments, run_length)

    return shuffled_vector

def generate_uniformly_distributed_partitions(total_duration, num_partitions):
    """
    Generate partitions where each partition size follows a uniform distribution 
    and the sum exactly equals total_duration.
    """
    if num_partitions == 1:
        return [total_duration]  # Only one partition, so it gets all the time

    # Step 1: Generate `num_partitions - 1` random breakpoints in [0, total_duration]
    cut_points = np.sort(np.random.choice(np.arange(1, total_duration), num_partitions - 1, replace=False))

    # Step 2: Compute partition sizes as the differences between consecutive breakpoints
    partitions = np.diff(np.concatenate(([0], cut_points, [total_duration]))).astype(int)

    # Step 3: Adjust rounding errors to ensure the exact sum
    while np.sum(partitions) != total_duration:
        diff = total_duration - partitions.sum()

        if diff > 0:
            idx = np.random.choice(len(partitions), size=diff, replace=True)
            for i in idx:
                partitions[i] += 1  # Increase partitions to match the total
        elif diff < 0:
            idx = np.random.choice(np.where(partitions > 1)[0], size=-diff, replace=True)
            for i in idx:
                partitions[i] -= 1  # Reduce partitions to match the total

    return partitions.tolist()

def create_labeled_segments(fixation_durations, non_fixation_durations):
    """
    Create labeled segments for fixation (1) and non-fixation (0) intervals.
    """
    fixation_labels = [(dur, 1) for dur in fixation_durations]
    non_fixation_labels = [(dur, 0) for dur in non_fixation_durations]
    return fixation_labels + non_fixation_labels

def construct_shuffled_vector(segments, run_length):
    """
    Construct a binary vector from shuffled fixation and non-fixation segments.
    """
    shuffled_vector = np.zeros(run_length, dtype=int)
    index = 0
    for duration, label in segments:
        if index >= run_length:
            logging.warning("Index exceeded run length during shuffle generation")
            break
        if label == 1:
            shuffled_vector[index: index + duration] = 1  # Fixation interval
        index += duration
    return shuffled_vector

def compute_shuffled_crosscorr_both_wrapper(pair):
    """Helper function to compute cross-correlations for shuffled vectors."""
    return fft_crosscorrelation_both(pair[0], pair[1])


# Plot difference between fix-vector crosscorr and shuffled vec crosscorr
def plot_fixation_crosscorr_minus_shuffled(inter_agent_behav_cross_correlation_df, monkeys_per_session_df, params, 
                                           group_by="session_name", plot_duration_seconds=60):
    """
    Plots the average (crosscorr - mean_shuffled) for both m1->m2 and m2->m1 
    across all runs, grouping either by session name or monkey pair.

    Parameters:
    - inter_agent_behav_cross_correlation_df: DataFrame containing cross-correlations.
    - monkeys_per_session_df: DataFrame containing session-wise monkey pairs.
    - params: Dictionary containing path parameters.
    - group_by: "session_name" or "monkey_pair" to determine averaging method.
    - plot_duration_seconds: Number of seconds to plot (default: 60s, assuming 1kHz sampling rate).
    """

    # Sampling rate (1kHz means 1000 time points per second)
    sample_rate = 1000  
    max_timepoints = plot_duration_seconds * sample_rate  

    # Merge session-wise monkey data into correlation dataframe
    merged_df = inter_agent_behav_cross_correlation_df.merge(monkeys_per_session_df, on="session_name")

    # Define monkey pair column
    merged_df["monkey_pair"] = merged_df["m1"] + "_" + merged_df["m2"]

    # Choose grouping variable
    group_column = "session_name" if group_by == "session_name" else "monkey_pair"

    # Group by session or monkey pair
    grouped = merged_df.groupby([group_column, "fixation_type"])

    # Compute mean and SEM of cross-correlation difference (trimming to `max_timepoints`)
    results = grouped.apply(compute_crosscorr_stats, max_timepoints=max_timepoints).reset_index()

    # Create plot directory
    today_date = datetime.today().strftime('%Y-%m-%d') + "_" + group_by
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_vector_crosscorrelations", today_date)
    os.makedirs(root_dir, exist_ok=True)

    # Get unique fixation types
    fixation_types = results["fixation_type"].unique()
    num_subplots = len(fixation_types)

    # Iterate over groups (sessions or monkey pairs)
    for group_value, group_df in tqdm(results.groupby(group_column), desc=f"Plotting for diff {group_by}"):
        fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4))

        if num_subplots == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one fixation type

        for ax, fixation_type in zip(axes, fixation_types):
            subset = group_df[group_df["fixation_type"] == fixation_type]
            if subset.empty:
                continue

            mean_m1_m2 = subset["mean_diff_m1_m2"].values[0]
            sem_m1_m2 = subset["sem_diff_m1_m2"].values[0]  # Updated to SEM
            mean_m2_m1 = subset["mean_diff_m2_m1"].values[0]
            sem_m2_m1 = subset["sem_diff_m2_m1"].values[0]  # Updated to SEM

            time_bins = np.arange(len(mean_m1_m2)) / sample_rate  # Convert to seconds

            ax.plot(time_bins, mean_m1_m2, label="m1 -> m2", color="blue")
            ax.fill_between(time_bins, mean_m1_m2 - sem_m1_m2, mean_m1_m2 + sem_m1_m2, color="blue", alpha=0.3)

            ax.plot(time_bins, mean_m2_m1, label="m2 -> m1", color="red")
            ax.fill_between(time_bins, mean_m2_m1 - sem_m2_m1, mean_m2_m1 + sem_m2_m1, color="red", alpha=0.3)

            ax.set_title(f"{fixation_type}")
            ax.set_xlabel("Time (seconds)")
            ax.legend()

        fig.suptitle(f"Fixation Cross-Correlation ({group_by}: {group_value})")
        fig.tight_layout()

        # Save plot
        plot_path = os.path.join(root_dir, f"{group_value}.png")
        plt.savefig(plot_path)
        plt.close()

    logger.info(f"Plots saved in {root_dir}")

def compute_crosscorr_stats(x, max_timepoints):
    """
    Computes mean and SEM of cross-correlation differences, ensuring all arrays are trimmed 
    to `max_timepoints` before stacking.
    
    Parameters:
    - x: DataFrame subset for a group.
    - max_timepoints: Maximum number of timepoints to include.
    
    Returns:
    - Pandas Series containing mean and SEM of (crosscorr - mean_shuffled).
    """
    crosscorr_m1_m2 = np.vstack([arr[:max_timepoints] for arr in x["crosscorr_m1_m2"].values])
    mean_shuffled_m1_m2 = np.vstack([arr[:max_timepoints] for arr in x["mean_shuffled_m1_m2"].values])
    crosscorr_m2_m1 = np.vstack([arr[:max_timepoints] for arr in x["crosscorr_m2_m1"].values])
    mean_shuffled_m2_m1 = np.vstack([arr[:max_timepoints] for arr in x["mean_shuffled_m2_m1"].values])

    return pd.Series({
        "mean_diff_m1_m2": np.mean(crosscorr_m1_m2 - mean_shuffled_m1_m2, axis=0),
        "sem_diff_m1_m2": sem(crosscorr_m1_m2 - mean_shuffled_m1_m2, axis=0, nan_policy='omit'),
        "mean_diff_m2_m1": np.mean(crosscorr_m2_m1 - mean_shuffled_m2_m1, axis=0),
        "sem_diff_m2_m1": sem(crosscorr_m2_m1 - mean_shuffled_m2_m1, axis=0, nan_policy='omit'),
    })


## Plot significant crosscorr time bins
def plot_significant_fixation_crosscorr_minus_shuffled(inter_agent_behav_cross_correlation_df, monkeys_per_session_df, params, 
                                                                group_by="session_name", plot_duration_seconds=90, alpha=0.05):
    """
    Plots the average (crosscorr - mean_shuffled) with only significant time bins colored,
    grouping either by session name or monkey pair. Uses parallel processing to speed up computation.

    Parameters:
    - inter_agent_behav_cross_correlation_df: DataFrame containing cross-correlations.
    - monkeys_per_session_df: DataFrame containing session-wise monkey pairs.
    - params: Dictionary containing path parameters and `num_cpus` for parallel processing.
    - group_by: "session_name" or "monkey_pair" to determine averaging method.
    - plot_duration_seconds: Number of seconds to plot (default: 90s, assuming 1kHz sampling rate).
    - alpha: Significance level for determining significant time bins.
    """

    # Sampling rate (1kHz means 1000 time points per second)
    sample_rate = 1000  
    max_timepoints = plot_duration_seconds * sample_rate  

    # Merge session-wise monkey data into correlation dataframe
    merged_df = inter_agent_behav_cross_correlation_df.merge(monkeys_per_session_df, on="session_name")

    # Define monkey pair column
    merged_df["monkey_pair"] = merged_df["m1"] + "_" + merged_df["m2"]

    # Choose grouping variable
    group_column = "session_name" if group_by == "session_name" else "monkey_pair"

    # Group by session or monkey pair
    grouped = list(merged_df.groupby([group_column, "fixation_type"]))

    # Parallelize the computation of significance stats
    num_cpus = params.get("num_cpus", -1)  # Use all available CPUs if not set

    logger.info("Computing bins with significant crosscorr compared to shuffled")
    results = Parallel(n_jobs=num_cpus, backend="loky")(
        delayed(compute_group_crosscorr_stats)(group_key, group_df, max_timepoints, alpha) for group_key, group_df in tqdm(grouped, desc="Computing statistics in parallel")
    )

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results, columns=["group_value", "fixation_type", "computed_stats"])

    # Extract dictionary columns back into separate columns
    stats_df = pd.concat([results_df.drop(columns=["computed_stats"]), results_df["computed_stats"].apply(pd.Series)], axis=1)

    # Create plot directory
    today_date = datetime.today().strftime('%Y-%m-%d') + "_" + group_by
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_vector_crosscorrelations_significant_parallel", today_date)
    os.makedirs(root_dir, exist_ok=True)

    # Get unique fixation types
    fixation_types = stats_df["fixation_type"].unique()
    num_subplots = len(fixation_types)

    # Iterate over groups (sessions or monkey pairs) for plotting
    for group_value, group_df in tqdm(stats_df.groupby("group_value"), desc=f"Plotting for {group_by}"):
        fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 3 * num_subplots), sharex=True)

        if num_subplots == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one fixation type

        for ax, fixation_type in zip(axes, fixation_types):
            subset = group_df[group_df["fixation_type"] == fixation_type]
            if subset.empty:
                continue

            mean_m1_m2 = subset["mean_diff_m1_m2"].values[0]
            significant_bins_m1_m2 = subset["significant_bins_m1_m2"].values[0]
            mean_m2_m1 = subset["mean_diff_m2_m1"].values[0]
            significant_bins_m2_m1 = subset["significant_bins_m2_m1"].values[0]

            time_bins = np.arange(len(mean_m1_m2)) / sample_rate  # Convert to seconds

            # Define colors
            sig_color_m1_m2, non_sig_color_m1_m2 = "blue", "lightblue"
            sig_color_m2_m1, non_sig_color_m2_m1 = "red", "lightcoral"

            # Check if there are significant bins
            any_sig_m1_m2 = np.any(significant_bins_m1_m2)
            any_sig_m2_m1 = np.any(significant_bins_m2_m1)

            # Plot non-significant bins first (thinner, lighter)
            ax.plot(time_bins, mean_m1_m2, color=non_sig_color_m1_m2, linewidth=0.75, alpha=0.5, label="m1 → m2 (non-sig)")
            ax.plot(time_bins, mean_m2_m1, color=non_sig_color_m2_m1, linewidth=0.75, alpha=0.5, label="m2 → m1 (non-sig)")

            # Mask for significant bins (only plot if they exist)
            if any_sig_m1_m2:
                sig_time_bins_m1_m2 = np.where(significant_bins_m1_m2, time_bins, np.nan)
                ax.plot(sig_time_bins_m1_m2, mean_m1_m2, color=sig_color_m1_m2, linewidth=1.5, label="m1 → m2 (sig)")

            if any_sig_m2_m1:
                sig_time_bins_m2_m1 = np.where(significant_bins_m2_m1, time_bins, np.nan)
                ax.plot(sig_time_bins_m2_m1, mean_m2_m1, color=sig_color_m2_m1, linewidth=1.5, label="m2 → m1 (sig)")

            # Add a note if no significant bins
            if not any_sig_m1_m2 and not any_sig_m2_m1:
                ax.set_title(f"{fixation_type} (No Significant Bins)")
            else:
                ax.set_title(f"{fixation_type}")

            ax.set_xlabel("Time (seconds)")
            ax.legend()

        fig.suptitle(f"Fixation Cross-Correlation (Significant Time Bins) ({group_by}: {group_value})")
        fig.tight_layout()

        # Save plot
        plot_path = os.path.join(root_dir, f"{group_value}.png")
        plt.savefig(plot_path)
        plt.close()

    logger.info(f"Plots saved in {root_dir}")

def compute_group_crosscorr_stats(group_key, group_df, max_timepoints, alpha):
    """
    Wrapper function to apply `compute_significant_crosscorr_stats_nonparametric` in parallel.
    
    Parameters:
    - group_key: Identifier for the group (e.g., session_name or monkey_pair).
    - group_df: DataFrame subset corresponding to that group.
    - max_timepoints: Maximum number of timepoints to include.
    - alpha: Significance threshold.

    Returns:
    - Tuple of (group_key, fixation_type, computed statistics).
    """
    result = compute_significant_crosscorr_stats_nonparametric(group_df, max_timepoints=max_timepoints, alpha=alpha)
    group_val, fix_type = group_key
    return (group_val, fix_type, result)

def compute_significant_crosscorr_stats_nonparametric(group_df, max_timepoints, alpha=0.05):
    """
    Computes mean cross-correlation differences and determines significant time bins 
    using the Wilcoxon signed-rank test instead of a t-test (for non-normal distributions).
    
    Parameters:
    - x: DataFrame subset for a group.
    - max_timepoints: Maximum number of timepoints to include.
    - alpha: Significance level for determining significant bins.
    
    Returns:
    - Pandas Series containing mean difference and significance mask for (crosscorr - mean_shuffled).
    """
    crosscorr_m1_m2 = np.vstack([arr[:max_timepoints] for arr in group_df["crosscorr_m1_m2"].values])
    mean_shuffled_m1_m2 = np.vstack([arr[:max_timepoints] for arr in group_df["mean_shuffled_m1_m2"].values])
    crosscorr_m2_m1 = np.vstack([arr[:max_timepoints] for arr in group_df["crosscorr_m2_m1"].values])
    mean_shuffled_m2_m1 = np.vstack([arr[:max_timepoints] for arr in group_df["mean_shuffled_m2_m1"].values])

    diff_m1_m2 = crosscorr_m1_m2 - mean_shuffled_m1_m2
    diff_m2_m1 = crosscorr_m2_m1 - mean_shuffled_m2_m1
    
    # Fully vectorized Wilcoxon test along time bins (axis=0)
    p_values_m1_m2 = wilcoxon(diff_m1_m2, alternative='two-sided', axis=0)[1]  # Extract only p-values
    p_values_m2_m1 = wilcoxon(diff_m2_m1, alternative='two-sided', axis=0)[1]

    # Determine significant time bins
    significant_bins_m1_m2 = p_values_m1_m2 < alpha
    significant_bins_m2_m1 = p_values_m2_m1 < alpha

    return pd.Series({
        "mean_diff_m1_m2": np.mean(diff_m1_m2, axis=0),
        "significant_bins_m1_m2": significant_bins_m1_m2,
        "mean_diff_m2_m1": np.mean(diff_m2_m1, axis=0),
        "significant_bins_m2_m1": significant_bins_m2_m1,
    })



# ** Call to main() **



if __name__ == "__main__":
    main()




