import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

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
    brain_region = unit["brain_region"]
    spike_times = np.array(unit["spike_ts"])
    rois = ["eyes", "non_eye_face", "object", "out_of_roi"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True)
    statistical_results = {}
    for idx, roi in enumerate(rois):
        transitions = ["eyes", "non_eye_face", "object", "out_of_roi"]
        spike_counts_per_transition = {trans: [] for trans in transitions}
        timeline = None
        for trans in transitions:
            spike_counts, timeline = ____compute_spiking_per_trial(
                roi, trans, session_behav_df, session_gaze_df, spike_times, params
            )
            if spike_counts:
                spike_counts_per_transition[trans] = spike_counts
        if timeline is not None:
            significant_bins = ___perform_anova_and_mark_plot(spike_counts_per_transition, timeline, axs[idx])
            if significant_bins > 0:
                statistical_results[roi] = significant_bins
        for trans in transitions:
            if spike_counts_per_transition[trans]:
                mean_spiking = np.mean(spike_counts_per_transition[trans], axis=0)
                axs[idx].plot(timeline[:-1], mean_spiking, label=f"{roi} → {trans}")
        axs[idx].set_title(f"Fixation on {roi}")
        axs[idx].legend()
    plt.suptitle(f"Fixation Transition Spiking for UUID {unit_uuid}")
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, f"fixation_transitions_{unit_uuid}.png"), dpi=100)
    plt.close(fig)
    return unit_uuid, brain_region, statistical_results


def ___perform_anova_and_mark_plot(spike_counts_per_transition, timeline, ax):
    """Performs ANOVA for each time bin across transition ROIs, applies FDR correction, and marks significant bins on the plot."""
    num_bins = len(timeline) - 1
    p_values = []
    for bin_idx in range(num_bins):
        data_per_transition = [np.array(spike_counts_per_transition[trans])[:, bin_idx] for trans in spike_counts_per_transition if spike_counts_per_transition[trans]]
        if len(data_per_transition) > 1:
            _, p_value = f_oneway(*data_per_transition)
            p_values.append(p_value)
        else:
            p_values.append(1.0)
    _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    significant_bins = np.where(corrected_p_values < 0.05)[0]
    for bin_idx in significant_bins:
        ax.axvline(timeline[bin_idx], color='red', linestyle='--', alpha=0.5)
    return len(significant_bins)


def ____compute_spiking_per_trial(roi, transition, session_behav_df, session_gaze_df, spike_times, params):
    """Computes spike counts per trial for a given transition type."""
    spike_counts_per_trial = []
    bin_size = params["neural_data_bin_size"]
    time_window = params['time_window_before_and_after_event_for_psth']
    timeline = np.arange(-time_window, time_window + bin_size, bin_size)
    for _, run_row in session_behav_df.iterrows():
        fixations = run_row["fixation_location"]
        run_number = run_row["run_number"]
        run_gaze_data = session_gaze_df[session_gaze_df["run_number"] == run_number]
        if run_gaze_data.empty:
            continue
        neural_timeline = run_gaze_data.iloc[0]["neural_timeline"]
        for fixation_start in run_row["fixation_start_stop"]:
            fixation_time = neural_timeline[fixation_start[0]]
            bins = np.linspace(fixation_time - time_window, fixation_time + time_window, int(2 * time_window / bin_size) + 1)
            spike_counts, _ = np.histogram(spike_times, bins=bins)
            spike_counts_per_trial.append(spike_counts / bin_size)
    return spike_counts_per_trial, timeline


if __name__ == "__main__":
    main()
