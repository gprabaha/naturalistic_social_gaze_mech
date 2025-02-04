import numpy as np
import pandas as pd
import os
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

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
        'is_grace': True,
        'recompute_fix_binary_vector': False,
        'recompute_crosscorr': False
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

    # Fix-related binary vectors
    fix_binary_vector_file = os.path.join(processed_data_dir, 'fix_binary_vector_df.pkl')
        
    if params.get('recompute_fix_binary_vector', False):
        logger.info("Generating fix-related binary vectors")
        fix_binary_vector_df = _generate_fixation_binary_vectors(eye_mvm_behav_df)
        fix_binary_vector_df.to_pickle(fix_binary_vector_file)
        logger.info(f"Fix-related binary vectors computed and saved to {fix_binary_vector_file}")
    else:
        logger.info("Loading precomputed fix-related binary vectors")
        fix_binary_vector_df = load_data.get_data_df(fix_binary_vector_file)

    # Cross-correlation computation
    inter_agent_cross_corr_file = os.path.join(processed_data_dir, 'inter_agent_behav_cross_correlation_df.pkl')

    if params.get('recompute_crosscorr', False):
        logger.info("Computing cross-correlations and shuffled statistics")
        inter_agent_behav_cross_correlation_df = _compute_crosscorr_and_shuffled_stats(
            fix_binary_vector_df, eye_mvm_behav_df, sigma=5, num_shuffles=50, num_cpus=cpu_count()
        )
        inter_agent_behav_cross_correlation_df.to_pickle(inter_agent_cross_corr_file)
        logger.info(f"Cross-correlations and shuffled statistics computed and saved to {inter_agent_cross_corr_file}")
    else:
        logger.info("Loading precomputed cross-correlations and shuffled statistics")
        inter_agent_behav_cross_correlation_df = load_data.get_data_df(inter_agent_cross_corr_file)

    _plot_fixation_crosscorr_minus_shuffled(inter_agent_behav_cross_correlation_df, monkeys_per_session_df, params, 
                        group_by="session_name", plot_duration_seconds=60)


# ** Sub-functions **


def _generate_fixation_binary_vectors(df):
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
        fixation_categories = __categorize_fixations(row["fixation_location"])

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

def __categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]


def _compute_crosscorr_and_shuffled_stats(df, eye_mvm_behav_df, sigma=5, num_shuffles=20, num_cpus=16):
    """
    Compute Gaussian-smoothed cross-correlation between m1 and m2's fixation behaviors.
    """
    results = []
    grouped = df.groupby(["session_name", "interaction_type", "run_number", "fixation_type"])
    
    for (session, interaction, run, fixation_type), group in tqdm(grouped, desc="Processing fix-type for runs"):
        if len(group) != 2:
            continue  # Skip if both m1 and m2 aren't present
        
        m1_row = group[group["agent"] == "m1"].iloc[0]
        m2_row = group[group["agent"] == "m2"].iloc[0]

        m1_vector = np.array(m1_row["binary_vector"])
        m2_vector = np.array(m2_row["binary_vector"])
        
        m1_smooth = gaussian_filter1d(m1_vector, sigma)
        m2_smooth = gaussian_filter1d(m2_vector, sigma)

        # Compute original cross-correlations
        crosscorr_m1_m2, crosscorr_m2_m1 = __fft_crosscorrelation_both(m1_smooth, m2_smooth)
        
        # Generate shuffled vectors
        m1_shuffled_vectors = __generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, num_shuffles)
        m2_shuffled_vectors = __generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, num_shuffles)
        
        # Parallel processing for shuffled cross-correlations
        num_processes = min(16, num_cpus)
        with Pool(num_processes) as pool:
            shuffled_crosscorrs = pool.map(
                __compute_shuffled_crosscorr_both_wrapper,
                [(gaussian_filter1d(m1_shuff, sigma), gaussian_filter1d(m2_shuff, sigma)) 
                 for m1_shuff, m2_shuff in zip(m1_shuffled_vectors, m2_shuffled_vectors)]
            )
        
        shuffled_crosscorrs_m1_m2 = np.array([s[0] for s in shuffled_crosscorrs])
        shuffled_crosscorrs_m2_m1 = np.array([s[1] for s in shuffled_crosscorrs])
        
        results.append({
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
        })
        
    return pd.DataFrame(results)

def __fft_crosscorrelation_both(x, y):
    """Compute cross-correlation using fftconvolve."""
    full_corr = fftconvolve(x, y[::-1], mode='full')

    n = len(x)
    mid = len(full_corr) // 2

    return full_corr[mid:mid + n], full_corr[:mid][::-1]

def __generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, num_shuffles=20):
    """
    Generate shuffled versions of a binary fixation vector by randomly redistributing fixation intervals.
    """
    row = eye_mvm_behav_df[(eye_mvm_behav_df["session_name"] == session) &
                            (eye_mvm_behav_df["interaction_type"] == interaction) &
                            (eye_mvm_behav_df["run_number"] == run)].iloc[0]
    
    run_length = row["run_length"]
    categories = __categorize_fixations(row["fixation_location"])
    fixation_start_stop = row["fixation_start_stop"]
    
    # Define which categories to include based on the requested fixation_type
    if fixation_type == "face":
        valid_categories = {"eyes", "non_eye_face"}
    else:
        valid_categories = {fixation_type}
    
    # Get only the fixations of the requested type(s)
    fixation_intervals = [
        (start, stop) for (start, stop), category in zip(fixation_start_stop, categories) if category in valid_categories
    ]
    fixation_durations = [stop - start + 1 for start, stop in fixation_intervals]
    total_fixation_duration = sum(fixation_durations)
    N = len(fixation_durations)
    
    # Compute total available non-fixation duration
    available_non_fixation_duration = run_length - total_fixation_duration
    
    shuffled_vectors = []
    for _ in range(num_shuffles):
        # Generate N+1 random non-fixation partitions that sum to available duration
        non_fixation_durations = np.random.multinomial(
            available_non_fixation_duration, 
            np.ones(N + 1) / (N + 1)
        )
        
        # Label fixation and non-fixation segments to ensure correct placement
        fixation_labels = [(dur, 1) for dur in fixation_durations]
        non_fixation_labels = [(dur, 0) for dur in non_fixation_durations]
        all_segments = fixation_labels + non_fixation_labels
        np.random.shuffle(all_segments)
        
        # Construct the shuffled binary vector
        shuffled_vector = np.zeros(run_length, dtype=int)
        index = 0
        for duration, label in all_segments:
            if index >= run_length:
                logging.warning("Index exceeded run length during shuffle generation")
                break
            if label == 1:
                shuffled_vector[index: index + duration] = 1  # Fixation interval
            index += duration
        
        shuffled_vectors.append(shuffled_vector)
    
    return shuffled_vectors

def __compute_shuffled_crosscorr_both_wrapper(pair):
    """Helper function to compute cross-correlations for shuffled vectors."""
    return __fft_crosscorrelation_both(pair[0], pair[1])



def _plot_fixation_crosscorr_minus_shuffled(inter_agent_behav_cross_correlation_df, monkeys_per_session_df, params, 
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

    # Compute mean and std of cross-correlation difference (only for first `max_timepoints` samples)
    results = grouped.apply(lambda x: pd.Series({
        "mean_diff_m1_m2": np.mean(np.vstack(x["crosscorr_m1_m2"].values)[:, :max_timepoints] - 
                                   np.vstack(x["mean_shuffled_m1_m2"].values)[:, :max_timepoints], axis=0),
        "std_diff_m1_m2": np.std(np.vstack(x["crosscorr_m1_m2"].values)[:, :max_timepoints] - 
                                 np.vstack(x["mean_shuffled_m1_m2"].values)[:, :max_timepoints], axis=0),
        "mean_diff_m2_m1": np.mean(np.vstack(x["crosscorr_m2_m1"].values)[:, :max_timepoints] - 
                                   np.vstack(x["mean_shuffled_m2_m1"].values)[:, :max_timepoints], axis=0),
        "std_diff_m2_m1": np.std(np.vstack(x["crosscorr_m2_m1"].values)[:, :max_timepoints] - 
                                 np.vstack(x["mean_shuffled_m2_m1"].values)[:, :max_timepoints], axis=0),
    })).reset_index()

    # Create plot directory
    today_date = datetime.today().strftime('%Y-%m-%d') + "_" + group_by
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_vector_crosscorrelations", today_date)
    os.makedirs(root_dir, exist_ok=True)

    # Get unique fixation types
    fixation_types = results["fixation_type"].unique()
    num_subplots = len(fixation_types)

    # Iterate over groups (sessions or monkey pairs)
    for group_value, group_df in tqdm(results.groupby(group_column), desc=f"Plotting for {group_by}"):
        fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4), sharey=True)

        if num_subplots == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one fixation type

        for ax, fixation_type in zip(axes, fixation_types):
            subset = group_df[group_df["fixation_type"] == fixation_type]
            if subset.empty:
                continue

            mean_m1_m2 = subset["mean_diff_m1_m2"].values[0]
            std_m1_m2 = subset["std_diff_m1_m2"].values[0]
            mean_m2_m1 = subset["mean_diff_m2_m1"].values[0]
            std_m2_m1 = subset["std_diff_m2_m1"].values[0]

            time_bins = np.arange(len(mean_m1_m2)) / sample_rate  # Convert to seconds

            ax.plot(time_bins, mean_m1_m2, label="m1 -> m2", color="blue")
            ax.fill_between(time_bins, mean_m1_m2 - std_m1_m2, mean_m1_m2 + std_m1_m2, color="blue", alpha=0.3)

            ax.plot(time_bins, mean_m2_m1, label="m2 -> m1", color="red")
            ax.fill_between(time_bins, mean_m2_m1 - std_m2_m1, mean_m2_m1 + std_m2_m1, color="red", alpha=0.3)

            ax.set_title(f"{fixation_type}")
            ax.set_xlabel("Time (seconds)")
            ax.legend()

        fig.suptitle(f"Fixation Cross-Correlation - Mean Shuffled ({group_by}: {group_value})")
        fig.tight_layout()

        # Save plot
        plot_path = os.path.join(root_dir, f"{group_value}_fix_crosscorr_diff_from_shuffled.png")
        plt.savefig(plot_path)
        plt.close()

    print(f"Plots saved in {root_dir}")


# ** Call to main() **


if __name__ == "__main__":
    main()
