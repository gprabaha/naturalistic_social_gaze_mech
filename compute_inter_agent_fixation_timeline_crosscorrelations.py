import numpy as np
import pandas as pd
import os
import logging
import random
import subprocess
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False

from datetime import datetime
from scipy.stats import sem
import random
from scipy.stats import wilcoxon, ttest_rel, shapiro

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
        'is_cluster': False,
        'prabaha_local': True,
        'is_grace': False,
        'use_parallel': True,
        'make_shuffle_stringent': True,
        'recompute_fix_binary_vector': False,
        'recompute_crosscorr': False
    }
    params = curate_data.add_num_cpus_to_params(params)
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

    num_cpus, threads_per_cpu = get_slurm_cpus_and_threads(params)
    
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
    if params.get('make_shuffle_stringent', False):
        inter_agent_cross_corr_file = os.path.join(processed_data_dir, 'inter_agent_behav_cross_correlation_df_stringent.pkl')
    else:
        inter_agent_cross_corr_file = os.path.join(processed_data_dir, 'inter_agent_behav_cross_correlation_df.pkl')

    if params.get('recompute_crosscorr', False):
        logger.info("Computing cross-correlations and shuffled statistics")
        inter_agent_behav_cross_correlation_df = compute_regular_and_shuffled_crosscorr(
            fix_binary_vector_df, eye_mvm_behav_df, params, sigma=2, num_shuffles=2000, num_cpus=num_cpus, threads_per_cpu=threads_per_cpu
        )
        inter_agent_behav_cross_correlation_df.to_pickle(inter_agent_cross_corr_file)
        logger.info(f"Cross-correlations and shuffled statistics computed and saved to {inter_agent_cross_corr_file}")
    else:
        logger.info("Loading precomputed cross-correlations and shuffled statistics")
        inter_agent_behav_cross_correlation_df = load_data.get_data_df(inter_agent_cross_corr_file)

    # Merge session-wise monkey data into correlation dataframe
    inter_agent_behav_cross_correlation_df = inter_agent_behav_cross_correlation_df.merge(monkeys_per_session_df, on="session_name")
    # Define monkey pair column
    inter_agent_behav_cross_correlation_df["monkey_pair"] = inter_agent_behav_cross_correlation_df["m1"] + "_" + inter_agent_behav_cross_correlation_df["m2"]

    # Plotting
    # logger.info("Generating cross-correlation plots for monkey pairs")
    # plot_fixation_crosscorrelations(inter_agent_behav_cross_correlation_df, params)
    # Plot summary of fixation cross-correlations
    logger.info("Generating summary of fixation cross-correlations")
    plot_fixation_crosscorrelation_summary(inter_agent_behav_cross_correlation_df, params)

    logger.info("Finished cross-correlation timecourse plot generation")



# ** Sub-functions **



def get_slurm_cpus_and_threads(params):
    """Returns the number of allocated CPUs and dynamically adjusts threads per CPU based on SLURM settings or local multiprocessing."""
    if params.get("is_cluster", False):
        # Get number of CPUs allocated by SLURM
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        slurm_cpus = int(slurm_cpus) if slurm_cpus else 1  # Default to 1 if not in SLURM
    else:
        # Get number of available CPUs using multiprocessing
        slurm_cpus = multiprocessing.cpu_count()
    # Default to 4 threads per CPU unless num_cpus is less than 4
    threads_per_cpu = 4 if slurm_cpus >= 4 else 1
    # Compute num_cpus by dividing total CPUs by threads per CPU
    num_cpus = max(1, slurm_cpus // threads_per_cpu)  # Ensure at least 1 CPU
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
def compute_regular_and_shuffled_crosscorr(fix_binary_vector_df, eye_mvm_behav_df, params, sigma=3, num_shuffles=100, num_cpus=16, threads_per_cpu=8):
    """
    Compute Gaussian-smoothed cross-correlation between m1 and m2's fixation behaviors.
    Runs in parallel if `params['use_parallel']` is True, otherwise runs in serial.
    """
    grouped = list(fix_binary_vector_df.groupby(["session_name", "interaction_type", "run_number", "fixation_type"]))
    if params.get("use_parallel", True):  # Default to parallel if flag is missing
        results = Parallel(n_jobs=num_cpus)(
            delayed(compute_crosscorr_for_group)(
                group_tuple, eye_mvm_behav_df, params, sigma, num_shuffles, threads_per_cpu
            ) for group_tuple in tqdm(grouped, desc="Processing groups")
        )
    else:
        results = [
            compute_crosscorr_for_group(group_tuple, eye_mvm_behav_df, params, sigma, num_shuffles, threads_per_cpu)
            for group_tuple in tqdm(grouped, desc="Processing groups")
        ]
    return pd.DataFrame([res for res in results if res])  # Filter out empty results


def compute_crosscorr_for_group(group_tuple, eye_mvm_behav_df, params, sigma, num_shuffles, num_threads):
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
    m1_shuffled_vectors = generate_shuffled_vectors(
        eye_mvm_behav_df, m1_vector, params, session, interaction, run, fixation_type, "m1", num_shuffles, num_threads)
    m2_shuffled_vectors = generate_shuffled_vectors(
        eye_mvm_behav_df, m2_vector, params, session, interaction, run, fixation_type, "m2", num_shuffles, num_threads)

    # plot_regular_fixation_vectors_and_example_shuffled_vectors(
    #     m1_vector, m2_vector, m1_shuffled_vectors, m2_shuffled_vectors, 
    #     params, session, interaction, run, fixation_type, num_shuffles_to_plot=5
    # )

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
    """Compute normalized cross-correlation using FFT, returning both positive and negative lags.
    If either x or y (or both) are all zeros, returns an array of zeros instead of NaNs.
    """
    n = len(x)
    # Handle cases where either x or y (or both) are all zeros
    if np.all(x == 0) or np.all(y == 0):
        return np.zeros(n), np.zeros(n)  # Return zero arrays of appropriate length
    # Subtract mean
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x = x - x_mean
    y = y - y_mean
    # Compute cross-correlation
    full_corr = fftconvolve(x, y[::-1], mode='full')
    # Normalize
    norm_factor = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) 
    if norm_factor == 0:  # Avoid division by zero
        return np.zeros(n), np.zeros(n)
    full_corr /= norm_factor
    # Extract positive and negative lags
    mid = len(full_corr) // 2
    return full_corr[mid:mid + n], full_corr[:mid][::-1]  # (Positive lags, Negative lags)


def generate_shuffled_vectors(
        eye_mvm_behav_df, agent_vector, params, session, interaction, run, fixation_type, agent, num_shuffles, num_threads):
    """
    Generate shuffled versions of a binary fixation vector by randomly redistributing fixation intervals.
    Runs in parallel if 'use_parallel' is True in params.
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
    if params.get("use_parallel", False):
        n_jobs = min(num_shuffles, num_threads) if params.get("is_cluster") else params["num_cpus"]
        shuffled_vectors = Parallel(n_jobs=n_jobs)(
            delayed(generate_single_shuffled_vector)(
                params, agent_vector, fixation_durations, available_non_fixation_duration, len(fixation_durations), run_length
            ) for _ in range(num_shuffles)
        )
    else:
        shuffled_vectors = [
            generate_single_shuffled_vector(
                params, agent_vector, fixation_durations, available_non_fixation_duration, len(fixation_durations), run_length
            ) for _ in range(num_shuffles)
        ]
    return shuffled_vectors

def generate_single_shuffled_vector(params, agent_vector, fixation_durations, available_non_fixation_duration, N, run_length):
    """
    Generates a single shuffled binary vector by redistributing fixation intervals.
    """
    if N == 0:
        return np.zeros(run_length, dtype=int)  # No fixations, return all zeros
    if params.get("make_shuffle_stringent", False):
        non_fixation_durations = generate_uniformly_distributed_partitions(available_non_fixation_duration, N + 1)
        all_segments = create_labeled_segments(fixation_durations, non_fixation_durations)
        shuffled_vector = construct_shuffled_vector(all_segments, run_length)
    else:
        shuffled_vector = np.random.permutation(agent_vector)
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
    Create labeled segments for fixation (1) and non-fixation (0) intervals, ensuring they alternate.
    """
    assert len(non_fixation_durations) == len(fixation_durations) + 1, "non_fixation_durations must have exactly one more element than fixation_durations."
    np.random.shuffle(fixation_durations)
    np.random.shuffle(non_fixation_durations)
    fixation_labels = [(dur, 1) for dur in fixation_durations]
    non_fixation_labels = [(dur, 0) for dur in non_fixation_durations]
    # Interleave non-fixation and fixation intervals
    all_segments = []
    for i in range(len(fixation_durations)):
        all_segments.append(non_fixation_labels[i])
        all_segments.append(fixation_labels[i])
    all_segments.append(non_fixation_labels[-1])  # Append the last non-fixation segment
    return all_segments

def construct_shuffled_vector(segments, run_length):
    """
    Construct a binary vector from shuffled fixation and non-fixation segments.
    """
    shuffled_vector = np.zeros(run_length, dtype=int)
    index = 0
    for duration, label in segments:
        if index >= run_length:
            logger.warning(f"Index {index} exceeded run length {run_length}")
            logger.info(f"All segments: {segments}")
            break
        if label == 1:
            shuffled_vector[index: index + duration] = 1  # Fixation interval
        index += duration
    return shuffled_vector

def compute_shuffled_crosscorr_both_wrapper(pair):
    """Helper function to compute cross-correlations for shuffled vectors."""
    return fft_crosscorrelation_both(pair[0], pair[1])



# ** Plotting Functions **

def plot_regular_fixation_vectors_and_example_shuffled_vectors(
    m1_vector, m2_vector, m1_shuffled_vectors, m2_shuffled_vectors,
    params, session, interaction, run, fixation_type, num_shuffles_to_plot=5):
    """
    Plots the original fixation vectors for M1 and M2 as broken bars in the first row,
    and separate subplots for shuffled fixation vectors in subsequent rows.
    Parameters:
    - m1_vector: np.array, binary fixation vector for M1
    - m2_vector: np.array, binary fixation vector for M2
    - m1_shuffled_vectors: np.array, 2D array of shuffled fixation vectors for M1
    - m2_shuffled_vectors: np.array, 2D array of shuffled fixation vectors for M2
    - params: dict, parameter dictionary containing 'root_data_dir'
    - session: str, session name
    - interaction: str, interaction type
    - run: int, run number
    - fixation_type: str, type of fixation
    - num_shuffles_to_plot: int, number of shuffled versions to plot
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    save_dir = os.path.join(params['root_data_dir'], "plots", "fixation_vectors", today_date)
    os.makedirs(save_dir, exist_ok=True)
    # Generate a unique filename including session, interaction, run, and fixation type
    filename = f"fixation_vectors_{session}_{interaction}_run{run}_{fixation_type}.pdf"
    filename = filename.replace(" ", "_")  # Replace spaces with underscores for compatibility
    save_path = os.path.join(save_dir, filename)
    # Define number of rows for plotting
    num_rows = num_shuffles_to_plot + 1  # +1 for the original
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(6, 1 * num_rows), sharex=True, sharey=True)

    def get_fixation_intervals(on_intervals):
        """Returns the start and stop indices of fixation periods."""
        if len(on_intervals) == 0:
            return [], []
        # Identify start and stop indices of fixation islands
        start_mask = np.insert(np.diff(on_intervals) > 1, 0, True)  # Always include first element
        stop_mask = np.append(np.diff(on_intervals) > 1, True)  # Always include last element
        starts = on_intervals[start_mask]
        stops = on_intervals[stop_mask]
        return starts, stops

    def plot_broken_bars(ax, vector, color="black", alpha=1.0, label=None):
        """Plots broken bars for binary fixation vectors, handling cases where no fixations exist."""
        on_intervals = np.where(vector == 1)[0]
        if len(on_intervals) > 0:
            starts, stops = get_fixation_intervals(on_intervals)
            bars = [(s, e - s + 1) for s, e in zip(starts, stops)]  # +1 to include last index
            ax.broken_barh(bars, (0, 0.8), facecolors=color, alpha=alpha, label=label)
        else:
            ax.text(0.5, 0.5, "No Fixations", ha="center", va="center", fontsize=8, color="gray")
    
    # Set title with session, interaction, run, and fixation type
    fig.suptitle(f"Fixation Vectors - Session: {session}, Interaction: {interaction}, Run: {run}, Type: {fixation_type}",
                 fontsize=12, fontweight="bold")
    axes[0, 0].set_title("M1 Fixation Vectors", fontsize=10)
    axes[0, 1].set_title("M2 Fixation Vectors", fontsize=10)
    # Plot original vectors
    plot_broken_bars(axes[0, 0], m1_vector, color="black", label="Original")
    plot_broken_bars(axes[0, 1], m2_vector, color="black", label="Original")
    # Ensure we do not request more shuffled samples than available
    num_shuffles_available = min(len(m1_shuffled_vectors), num_shuffles_to_plot)
    m1_random_shuffles = random.sample(range(len(m1_shuffled_vectors)), num_shuffles_available)
    m2_random_shuffles = random.sample(range(len(m2_shuffled_vectors)), num_shuffles_available)
    # Plot shuffled vectors
    for i, idx in enumerate(m1_random_shuffles):
        plot_broken_bars(axes[i+1, 0], m1_shuffled_vectors[idx], color=f"C{i}", alpha=0.6, label=f"Shuffle {i+1}")
    for i, idx in enumerate(m2_random_shuffles):
        plot_broken_bars(axes[i+1, 1], m2_shuffled_vectors[idx], color=f"C{i}", alpha=0.6, label=f"Shuffle {i+1}")
    # Formatting for better readability
    for i, ax_row in enumerate(axes):
        for ax in ax_row:
            ax.set_ylabel(f"Shuffle {i}" if i > 0 else "Original", fontsize=8)
            ax.set_yticks([])
            # Only add legend if data exists
            if len(ax.patches) > 0:
                ax.legend(loc="upper right", fontsize=6)
    axes[-1, 0].set_xlabel("Time (ms)", fontsize=10)
    axes[-1, 1].set_xlabel("Time (ms)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.savefig(save_path, format="pdf")
    plt.close()


def plot_fixation_crosscorrelations(inter_agent_behav_cross_correlation_df, params, max_time=60):
    max_time = int(max_time * 1000)  # Convert seconds to milliseconds (assuming 1KHz sampling rate)
    group_by = "monkey_pair"
    today_date = f"{datetime.today().strftime('%Y-%m-%d')}_{group_by}" + ("_stringent" if params.get('make_shuffle_stringent', False) else "")
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_vector_crosscorrelations_updated", today_date)
    os.makedirs(root_dir, exist_ok=True)
    logger.info(f"Processing cross-correlations, saving results to {root_dir}")
    grouped = inter_agent_behav_cross_correlation_df.groupby(group_by)
    for monkey_pair, group_df in grouped:
        logger.info(f"Processing monkey pair: {monkey_pair}")
        fixation_types = group_df['fixation_type'].unique()
        num_fixations = len(fixation_types)
        fig, axes = plt.subplots(num_fixations, 1, figsize=(12, 4 * num_fixations), sharex=True)
        if num_fixations == 1:
            axes = [axes]
        for ax, fixation in zip(axes, fixation_types):
            logger.info(f"Processing fixation type: {fixation}")
            subset = group_df[group_df['fixation_type'] == fixation]
            mean_crosscorr_m1_m2, mean_crosscorr_m2_m1, max_val, significant_bins_m1_m2, significant_bins_m2_m1, significant_diff_bins \
                = compute_crosscorr_means_and_stats(subset, max_time)
            time_bins = np.linspace(0, max_time / 1000, max_time)  # Convert time bins to seconds
            ax.plot(time_bins, mean_crosscorr_m1_m2, label='m1->m2', color='blue', alpha=0.5)
            ax.plot(time_bins, mean_crosscorr_m2_m1, label='m2->m1', color='red', alpha=0.5)
            
            ax.scatter(time_bins[significant_bins_m1_m2], mean_crosscorr_m1_m2[significant_bins_m1_m2],
                color='blue', s=20, label='sig m1->m2', linewidth=1.5)
            ax.scatter(time_bins[significant_bins_m2_m1], mean_crosscorr_m2_m1[significant_bins_m2_m1],
                color='red', s=20, label='sig m2->m1', linewidth=1.5)
            ax.scatter(time_bins[significant_diff_bins], np.full_like(time_bins[significant_diff_bins], max_val),
                color='black', marker='*', s=25, label='m1-m2 diff')
            
            ax.set_title(f"Monkey Pair: {monkey_pair} | Fixation: {fixation}")
            ax.legend()
            ax.set_ylabel("Cross-correlation")
        ax.set_xlabel("Time (seconds)")
        plt.tight_layout()
        save_path = os.path.join(root_dir, f"{monkey_pair}_crosscorr_plots.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved plot for {monkey_pair} at {save_path}")
    logger.info(f"All plots saved in {root_dir}")

def compute_crosscorr_means_and_stats(subset, max_time):
    crosscorr_m1_m2 = np.array([x[:max_time] for x in subset['crosscorr_m1_m2']])
    crosscorr_m2_m1 = np.array([x[:max_time] for x in subset['crosscorr_m2_m1']])
    mean_shuffled_m1_m2 = np.array([x[:max_time] for x in subset['mean_shuffled_m1_m2']])
    mean_shuffled_m2_m1 = np.array([x[:max_time] for x in subset['mean_shuffled_m2_m1']])
    mean_crosscorr_m1_m2 = np.mean(crosscorr_m1_m2, axis=0)
    mean_crosscorr_m2_m1 = np.mean(crosscorr_m2_m1, axis=0)
    max_val = max(np.max(mean_crosscorr_m1_m2), np.max(mean_crosscorr_m2_m1)) * 1.05
    
    def extract_p_value(x, y):
        return wilcoxon(x, y, alternative='two-sided')[1]  # Extract only the p-value
    
    p_values_m1_m2 = Parallel(n_jobs=-1)(delayed(extract_p_value)(crosscorr_m1_m2[:, i], mean_shuffled_m1_m2[:, i]) for i in range(max_time))
    p_values_m2_m1 = Parallel(n_jobs=-1)(delayed(extract_p_value)(crosscorr_m2_m1[:, i], mean_shuffled_m2_m1[:, i]) for i in range(max_time))
    significant_bins_m1_m2 = np.array(p_values_m1_m2) < 0.05
    significant_bins_m2_m1 = np.array(p_values_m2_m1) < 0.05
    diff_p_values = Parallel(n_jobs=-1)(delayed(extract_p_value)(crosscorr_m1_m2[:, i], crosscorr_m2_m1[:, i]) for i in range(max_time))
    significant_diff_bins = np.array(diff_p_values) < 0.05
    return mean_crosscorr_m1_m2, mean_crosscorr_m2_m1, max_val, significant_bins_m1_m2, significant_bins_m2_m1, significant_diff_bins


def plot_fixation_crosscorrelation_summary(inter_agent_behav_cross_correlation_df, params, max_time=30):
    """
    Plots the summary of fixation cross-correlations for monkey pairs.
    - Identifies monkey pairs where m1->m2 face cross-correlation is greater than m2->m1 in the first 10s.
    - Plots face and out-of-ROI fixation cross-correlations for these monkey pairs.
    - Highlights significant cross-correlation bins and significant differences.
    """
    max_time = int(max_time * 1000)  # Convert seconds to milliseconds (assuming 1KHz sampling rate)
    first_10_sec = int(10 * 1000)
    group_by = "monkey_pair"
    today_date = f"{datetime.today().strftime('%Y-%m-%d')}" + ("_stringent" if params.get('make_shuffle_stringent', False) else "")
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_vector_crosscorrelations_summary", today_date)
    os.makedirs(root_dir, exist_ok=True)
    logger.info(f"Processing cross-correlation summaries; saving results to {root_dir}")
    grouped = inter_agent_behav_cross_correlation_df.groupby(group_by)
    pairs_m1_greater = []
    pairs_m2_greater = []
    handles = []
    labels = []
    for monkey_pair, group_df in grouped:
        subset = group_df[group_df['fixation_type'] == 'face']
        if not subset.empty:
            mean_crosscorr_m1_m2, mean_crosscorr_m2_m1, _ = compute_mean_crosscorr(subset, max_time)
            if np.mean(mean_crosscorr_m1_m2[:first_10_sec]) > np.mean(mean_crosscorr_m2_m1[:first_10_sec]):
                pairs_m1_greater.append(group_df)
            else:
                pairs_m2_greater.append(group_df)
    pairs_m1_greater = pd.concat(pairs_m1_greater) if pairs_m1_greater else pd.DataFrame()
    pairs_m2_greater = pd.concat(pairs_m2_greater) if pairs_m2_greater else pd.DataFrame()
    conditions_to_plot = ['m1_greater', 'm2_greater']
    rois_to_plot = ['face', 'out_of_roi']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, condition in zip(axes, conditions_to_plot):
        for roi in rois_to_plot:
            subset = pairs_m1_greater if condition == 'm1_greater' else pairs_m2_greater
            subset = subset[subset['fixation_type'] == roi]
            if subset.empty:
                continue
            plot_colors = ['#FF4500', '#008B8B'] if roi == 'face' else ['black', 'gray']
            mean_crosscorr_m1_m2, mean_crosscorr_m2_m1, max_val = compute_mean_crosscorr(subset, max_time)
            significant_bins_m1_m2, significant_bins_m2_m1, significant_diff_bins = statistically_compare_crosscorrelations(subset, max_time)
            time_bins = np.linspace(0, max_time / 1000, max_time)
            line1, = ax.plot(time_bins, mean_crosscorr_m1_m2, label=f'm1→m2 ({roi})', color=plot_colors[0], alpha=0.5, linewidth=1)
            line2, = ax.plot(time_bins, mean_crosscorr_m2_m1, label=f'm2→m1 ({roi})', color=plot_colors[1], alpha=0.5, linewidth=1)
            mean_crosscorr_m1_m2_highlight = np.full_like(mean_crosscorr_m1_m2, np.nan)
            mean_crosscorr_m2_m1_highlight = np.full_like(mean_crosscorr_m2_m1, np.nan)
            mean_crosscorr_m1_m2_highlight[significant_bins_m1_m2] = mean_crosscorr_m1_m2[significant_bins_m1_m2]
            mean_crosscorr_m2_m1_highlight[significant_bins_m2_m1] = mean_crosscorr_m2_m1[significant_bins_m2_m1]
            ax.plot(time_bins, mean_crosscorr_m1_m2_highlight, color=plot_colors[0], linewidth=2)
            ax.plot(time_bins, mean_crosscorr_m2_m1_highlight, color=plot_colors[1], linewidth=2)
            if roi == 'face':
                ax.scatter(time_bins[significant_diff_bins], np.full_like(time_bins[significant_diff_bins], max_val),
                    color='black', marker='.', label=f'm1-m2 diff ({roi})', s=25)
            if condition == 'm1_greater':
                handles.extend([line1, line2])
                labels.extend([f'm1→m2 ({roi})', f'm2→m1 ({roi})'])
        ax.set_title("m1→m2 > m2→m1 pairs" if condition == 'm1_greater' else "m2→m1 > m1→m2 pairs")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Cross-correlation")
    fig.legend(handles, labels, loc='best')
    plt.tight_layout()
    save_path = os.path.join(root_dir, "fixation_crosscorr_summary.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.close(fig)
    logger.info(f"All plots saved in {root_dir}")

def compute_mean_crosscorr(subset, max_time):
    """
    Computes the mean cross-correlation values for m1->m2 and m2->m1.
    """
    crosscorr_m1_m2 = np.mean(np.vstack(subset['crosscorr_m1_m2'].apply(lambda x: x[:max_time])), axis=0)
    crosscorr_m2_m1 = np.mean(np.vstack(subset['crosscorr_m2_m1'].apply(lambda x: x[:max_time])), axis=0)
    max_val = max(np.max(crosscorr_m1_m2), np.max(crosscorr_m2_m1)) * 1.05
    return crosscorr_m1_m2, crosscorr_m2_m1, max_val

def statistically_compare_crosscorrelations(subset, max_time):
    """
    Performs statistical comparisons between cross-correlations and shuffled distributions.
    """
    crosscorr_m1_m2 = np.array([x[:max_time] for x in subset['crosscorr_m1_m2']])
    crosscorr_m2_m1 = np.array([x[:max_time] for x in subset['crosscorr_m2_m1']])
    mean_shuffled_m1_m2 = np.array([x[:max_time] for x in subset['mean_shuffled_m1_m2']])
    mean_shuffled_m2_m1 = np.array([x[:max_time] for x in subset['mean_shuffled_m2_m1']])
    
    def extract_p_value(x, y):
        return wilcoxon(x, y, alternative='two-sided')[1]
    
    p_values_m1_m2 = Parallel(n_jobs=-1)(delayed(extract_p_value)(crosscorr_m1_m2[:, i], mean_shuffled_m1_m2[:, i]) for i in range(max_time))
    p_values_m2_m1 = Parallel(n_jobs=-1)(delayed(extract_p_value)(crosscorr_m2_m1[:, i], mean_shuffled_m2_m1[:, i]) for i in range(max_time))
    diff_p_values = Parallel(n_jobs=-1)(delayed(extract_p_value)(crosscorr_m1_m2[:, i], crosscorr_m2_m1[:, i]) for i in range(max_time))
    return np.array(p_values_m1_m2) < 0.05, np.array(p_values_m2_m1) < 0.05, np.array(diff_p_values) < 0.05








# ** Call to main() **



if __name__ == "__main__":
    main()




