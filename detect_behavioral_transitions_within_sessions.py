import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
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
        'smooth_spike_counts': True
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
    for agent in tqdm(["m1", "m2"], desc=f"Processing agents in {session_name}"):
        agent_behav_df = session_behav_df[session_behav_df["agent"] == agent]
        transition_probs = __compute_fixation_transition_probabilities(agent_behav_df)
        print(f"Transition probabilities for {agent} in session {session_name}:")
        print(transition_probs)
        pdb.set_trace()
        transition_probs.to_csv(os.path.join(session_dir, f"transition_probabilities_{agent}.csv"))
    for _, unit in tqdm(session_spike_df.iterrows(), total=len(session_spike_df), desc=f"Processing units in {session_name}"):
        __plot_fixation_transition_spiking(unit, session_behav_df, session_gaze_df, session_dir, params)


def __compute_fixation_transition_probabilities(agent_behav_df):
    transitions = []
    for _, run_df in tqdm(agent_behav_df.groupby("run_number"), desc="Processing runs"):
        fixation_sequences = run_df["fixation_location"].values[0]
        categorized_fixations = [
            "eye" if {"face", "eyes_nf"}.issubset(set(fixes)) else
            "non_eye_face" if "face" in set(fixes) else
            "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
            for fixes in fixation_sequences
        ]
        transitions.extend(zip(categorized_fixations[:-1], categorized_fixations[1:]))
    return pd.DataFrame(transitions, columns=["from", "to"]).value_counts(normalize=True).reset_index(name="probability")


def __plot_fixation_transition_spiking(unit, session_behav_df, session_gaze_df, session_dir, params):
    unit_uuid = unit["unit_uuid"]
    spike_times = np.array(unit["spike_ts"])
    rois = ["eye", "non_eye_face", "object", "out_of_roi"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True)
    for idx, roi in enumerate(rois):
        transitions = ["eye", "non_eye_face", "object", "out_of_roi"]
        for trans in transitions:
            mean_activity, timeline = ____compute_mean_spiking_for_transition(roi, trans, session_behav_df, session_gaze_df, spike_times, params)
            if mean_activity is not None:
                axs[idx].plot(timeline[:-1], mean_activity, label=f"{roi} → {trans}")
        axs[idx].set_title(f"Fixation on {roi}")
        axs[idx].legend()
    plt.suptitle(f"Fixation Transition Spiking for UUID {unit_uuid}")
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, f"fixation_transitions_{unit_uuid}.png"), dpi=300)
    plt.close(fig)


def ____compute_mean_spiking_for_transition(roi, transition, session_behav_df, session_gaze_df, spike_times, params):
    direction_column = "fixation_location"
    mean_activity = []
    bin_size = params["neural_data_bin_size"]
    time_window = 1
    timeline = np.arange(-time_window, time_window + bin_size, bin_size)
    for _, run_row in session_behav_df.iterrows():
        fixations = run_row[direction_column]
        run_number = run_row["run_number"]
        relevant_fixations = [
            fixation_idx for fixation_idx, labels in enumerate(fixations[:-1])
            if (roi in labels and transition in fixations[fixation_idx + 1])
        ]
        if not relevant_fixations:
            continue
        run_gaze_data = session_gaze_df[session_gaze_df["run_number"] == run_number]
        if run_gaze_data.empty:
            continue
        neural_timeline = run_gaze_data.iloc[0]["neural_timeline"]
        for fixation_idx in relevant_fixations:
            fixation_start = run_row["fixation_start_stop"][fixation_idx][0]
            if fixation_start >= len(neural_timeline):
                continue
            fixation_time = neural_timeline[fixation_start]
            bins = np.linspace(
                fixation_time - time_window,
                fixation_time + time_window,
                int(2 * time_window / bin_size) + 1
            )
            spike_counts, _ = np.histogram(spike_times, bins=bins)
            spike_counts = spike_counts / bin_size  # Convert to firing rate
            if params["smooth_spike_counts"]:
                spike_counts = np.convolve(spike_counts, np.ones(5)/5, mode='same')
            mean_activity.append(spike_counts)
    if len(mean_activity) > 0:
        mean_activity = np.mean(mean_activity, axis=0)
        return mean_activity, timeline
    return None, None


if __name__ == "__main__":
    main()
