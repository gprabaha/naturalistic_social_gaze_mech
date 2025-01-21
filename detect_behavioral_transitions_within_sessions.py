import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

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
    _compute_transition_probabilities_and_spiking(eye_mvm_behav_df, spike_times_df, sparse_nan_removed_sync_gaze_df, params)


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


def _compute_transition_probabilities_and_spiking(eye_mvm_behav_df, spike_times_df, sparse_nan_removed_sync_gaze_df, params):
    logger.info("Computing transition probabilities and spiking responses")
    today_date = datetime.now().strftime("%Y-%m-%d")
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_transition_spiking", today_date)
    os.makedirs(root_dir, exist_ok=True)
    for session_name, session_behav_df in tqdm(eye_mvm_behav_df.groupby("session_name"), desc="Processing Sessions"):
        session_spike_df = spike_times_df[spike_times_df["session_name"] == session_name]
        session_gaze_df = sparse_nan_removed_sync_gaze_df[sparse_nan_removed_sync_gaze_df["session_name"] == session_name]
        session_dir = os.path.join(root_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        __process_session(session_name, session_behav_df, session_spike_df, session_gaze_df, session_dir, params)


def __process_session(session_name, session_behav_df, session_spike_df, session_gaze_df, session_dir, params):
    logger.info(f"Processing session {session_name}")
    for agent in ["m1", "m2"]:
        agent_behav_df = session_behav_df[session_behav_df["agent"] == agent]
        transition_probs = __compute_fixation_transition_probabilities(agent_behav_df)
        __plot_transition_matrix(transition_probs, session_dir, agent)
        transition_probs.to_csv(os.path.join(session_dir, f"transition_probabilities_{agent}.csv"))
    statistical_summary = []
    for _, unit in tqdm(session_spike_df.iterrows(), total=len(session_spike_df), desc=f"Processing units in {session_name}"):
        unit_uuid, brain_region, significant_results = __plot_fixation_transition_spiking(
            unit, session_behav_df, session_gaze_df, session_dir, params
        )
        if significant_results:
            statistical_summary.append((unit_uuid, brain_region, significant_results))
    if statistical_summary:
        logger.info(f"Summary of significant effects in session {session_name}:")
        for unit_uuid, brain_region, sig_results in statistical_summary:
            for roi, count in sig_results.items():
                logger.info(f"Unit {unit_uuid} ({brain_region}): {roi} transition significant in {count} bins.")
    return statistical_summary


def __compute_fixation_transition_probabilities(agent_behav_df):
    transitions = []
    for _, run_df in agent_behav_df.groupby("run_number"):
        fixation_sequences = run_df["fixation_location"].values[0]
        categorized_fixations = [
            "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
            "non_eye_face" if "face" in set(fixes) else
            "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
            for fixes in fixation_sequences
        ]
        transitions.extend(zip(categorized_fixations[:-1], categorized_fixations[1:]))
    return pd.DataFrame(transitions, columns=["from", "to"]).value_counts(normalize=True).reset_index(name="probability")


def __plot_transition_matrix(transition_df, session_dir, agent):
    pivot_table = transition_df.pivot(index="from", columns="to", values="probability").fillna(0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".4f", linewidths=0.5)
    plt.title(f"Fixation Transition Probabilities ({agent})")
    plt.xlabel("To Fixation")
    plt.ylabel("From Fixation")
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, f"fixation_transition_matrix_{agent}.png"), dpi=100)
    plt.close()


def __plot_fixation_transition_spiking(unit, session_behav_df, session_gaze_df, session_dir, params):
    """Plots fixation transition spiking and runs statistical tests."""
    unit_uuid = unit["unit_uuid"]
    brain_region = unit["region"]
    spike_times = np.array(unit["spike_ts"])
    rois = ["eyes", "non_eye_face", "object", "out_of_roi"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True)
    statistical_results = {}
    for idx, roi in enumerate(rois):
        transitions = ["eyes", "non_eye_face", "object", "out_of_roi"]
        spike_data = []  # List to store (spike_counts, transition_label) tuples
        timeline = None
        for trans in transitions:
            trial_spike_counts, timeline = ____compute_spiking_per_trial(
                roi, trans, session_behav_df, session_gaze_df, spike_times, params
            )
            if trial_spike_counts:
                spike_data.extend(trial_spike_counts)  # Append trial data
        if timeline is not None:
            significant_bins = ___perform_anova_and_mark_plot(spike_data, timeline, axs[idx])
            if significant_bins > 0:
                statistical_results[roi] = significant_bins
        # Compute mean and plot it
        if spike_data:
            spike_df = pd.DataFrame(spike_data, columns=["spike_counts", "transition"])
            mean_spiking = {}
            for transition, data in spike_df.groupby("transition")["spike_counts"]:
                spike_matrix = np.array(data.tolist())  # Convert list of arrays to a 2D matrix
                mean_spiking[transition] = np.mean(spike_matrix, axis=0)  # Compute mean across trials
            for transition, mean_values in mean_spiking.items():
                axs[idx].plot(timeline[:-1], mean_values, label=transition)
        axs[idx].set_title(f"Fixation on {roi}")
        axs[idx].legend()
    plt.suptitle(f"Fixation Transition Spiking for UUID {unit_uuid}")
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, f"fixation_transitions_{unit_uuid}.png"), dpi=100)
    plt.close(fig)
    return unit_uuid, brain_region, statistical_results


def ____compute_spiking_per_trial(roi, transition, session_behav_df, session_gaze_df, spike_times, params):
    """Computes spike counts per trial for a given transition type."""
    spike_counts_per_trial = []
    bin_size = params["neural_data_bin_size"]
    time_window = params['time_window_before_and_after_event_for_psth']
    timeline = np.arange(-time_window, time_window + bin_size, bin_size)
    # Define smoothing kernel if needed
    smoothing_kernel = np.ones(5) / 5 if params["smooth_spike_counts"] else None
    for _, run_row in session_behav_df.iterrows():
        fixations = run_row["fixation_location"]
        # Categorize fixations
        categorized_fixations = [
            "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
            "non_eye_face" if "face" in set(fixes) else
            "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
            for fixes in fixations
        ]
        run_number = run_row["run_number"]
        # Identify relevant fixations that transition from `roi` to `transition`
        relevant_fixations = [
            fixation_idx for fixation_idx, fix_label in enumerate(categorized_fixations[:-1])
            if (fix_label == roi and categorized_fixations[fixation_idx + 1] == transition)
        ]
        if not relevant_fixations:
            continue  # Skip if no relevant fixations in this run
        run_gaze_data = session_gaze_df[session_gaze_df["run_number"] == run_number]
        if run_gaze_data.empty:
            continue
        neural_timeline = run_gaze_data.iloc[0]["neural_timeline"]
        for fixation_idx in relevant_fixations:
            fixation_start = run_row["fixation_start_stop"][fixation_idx][0]
            if fixation_start >= len(neural_timeline):
                continue  # Skip if the fixation start index is out of bounds
            fixation_time = neural_timeline[fixation_start]
            # Define time bins around fixation onset
            bins = np.linspace(
                fixation_time - time_window,
                fixation_time + time_window,
                int(2 * time_window / bin_size) + 1
            ).ravel()
            # Compute spike histogram
            spike_counts, _ = np.histogram(spike_times, bins=bins)
            spike_counts = spike_counts / bin_size  # Convert to firing rate
            # Apply smoothing if enabled
            if params["smooth_spike_counts"]:
                spike_counts = np.convolve(spike_counts, smoothing_kernel, mode='same')
            # Append to trial list with the correct transition label
            spike_counts_per_trial.append((spike_counts, f"{roi} → {transition}"))
    return spike_counts_per_trial, timeline


def ___perform_anova_and_mark_plot(spike_data, timeline, ax):
    """Performs ANOVA for each time bin across transition ROIs, applies FDR correction, and marks significant bins on the plot."""
    num_bins = len(timeline) - 1
    spike_df = pd.DataFrame(spike_data, columns=["spike_counts", "transition"])
    p_values = []
    for bin_idx in range(num_bins):
        bin_data = spike_df["spike_counts"].apply(lambda x: x[bin_idx])  # Extract bin-wise spike counts
        groups = spike_df["transition"]
        # ANOVA across transitions
        unique_groups = groups.unique()
        spike_lists = [bin_data[groups == grp].tolist() for grp in unique_groups]
        if len(spike_lists) > 1:
            _, p_value = f_oneway(*spike_lists)
            p_values.append(p_value)
        else:
            p_values.append(1.0)
    # Apply FDR correction
    _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    # Mark significant bins
    significant_bins = np.where(corrected_p_values < 0.05)[0]
    for bin_idx in significant_bins:
        ax.axvline(timeline[bin_idx], color='red', linestyle='--', alpha=0.5)
    return len(significant_bins)


if __name__ == "__main__":
    main()
