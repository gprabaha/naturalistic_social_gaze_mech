import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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
        'num_cpus': min(8, cpu_count())
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
    session_tasks = []
    for session_name, session_behav_df in eye_mvm_behav_df.groupby("session_name"):
        session_spike_df = spike_times_df[spike_times_df["session_name"] == session_name]
        session_gaze_df = sparse_nan_removed_sync_gaze_df[sparse_nan_removed_sync_gaze_df["session_name"] == session_name]
        session_dir = os.path.join(root_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        session_tasks.append((session_name, session_behav_df, session_spike_df, session_gaze_df, session_dir, params))
    logger.info(f"Running parallel processing with {params['num_cpus']} CPUs")
    with Pool(processes=params['num_cpus']) as pool:
        list(tqdm(pool.imap(__process_session, session_tasks), total=len(session_tasks), desc="Sessions"))


def __process_session(task):
    session_name, session_behav_df, session_spike_df, session_gaze_df, session_dir, params = task
    logger.info(f"Processing session {session_name}")
    for agent in ["m1", "m2"]:
        agent_behav_df = session_behav_df[session_behav_df["agent"] == agent]
        transition_probs = __compute_fixation_transition_probabilities(agent_behav_df)
        transition_probs.to_csv(os.path.join(session_dir, f"transition_probabilities_{agent}.csv"))
    for _, unit in session_spike_df.iterrows():
        __plot_fixation_transition_spiking(unit, session_behav_df, session_gaze_df, session_dir, params)


def __compute_fixation_transition_probabilities(agent_behav_df):
    transitions = []
    for _, run_df in agent_behav_df.groupby("run_number"):
        fixation_sequences = run_df["fixation_location"].tolist()
        categorized_fixations = [
            "eye" if "eyes_nf" in fix and "face" in fix else
            "non-eye-face" if "face" in fix else
            "object" if "object" in fix else "out_of_roi" 
            for fix in fixation_sequences
        ]
        transitions.extend(zip(categorized_fixations[:-1], categorized_fixations[1:]))
    return pd.DataFrame(transitions, columns=["from", "to"]).value_counts(normalize=True).reset_index(name="probability")


def __plot_fixation_transition_spiking(unit, session_behav_df, session_gaze_df, session_dir, params):
    unit_uuid = unit["unit_uuid"]
    spike_times = np.array(unit["spike_ts"])
    rois = ["eye", "non-eye-face", "object", "out_of_roi"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True)
    for idx, roi in enumerate(rois):
        transitions = ["eye", "non-eye-face", "object", "out_of_roi"]
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


if __name__ == "__main__":
    main()
