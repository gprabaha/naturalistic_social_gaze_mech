import numpy as np
import pandas as pd
import os
import logging
from itertools import chain
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import load_data
import curate_data
import pdb



# Configure logging to output to both file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)



def main():
    """
    Main function to load data, initialize parameters, and generate mean spiking activity plots.
    """
    logger.info("Starting the main function")
    params = _initialize_params()
    sparse_nan_removed_sync_gaze_data_df_filepath = os.path.join(
        params['processed_data_dir'], 'sparse_nan_removed_sync_gaze_data_df.pkl'
    )
    logger.info("Loading sparse_nan_removed_sync_gaze_df")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df_file_path = os.path.join(
        params['processed_data_dir'], 'eye_mvm_behav_df.pkl'
    )
    logger.info("Loading eye_mvm_behav_df")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_file_path = os.path.join(
        params['processed_data_dir'], 'spike_times_df.pkl'
    )
    logger.info("Loading spike times data")
    spike_times_df = load_data.get_data_df(spike_times_file_path)
    _plot_mean_behavioral_spiking_response(
        eye_mvm_behav_df, spike_times_df, sparse_nan_removed_sync_gaze_df, params
    )



def _initialize_params():
    """
    Initialize parameters required for the script and load data paths.
    Returns:
        dict: Initialized parameters.
    """
    logger.info("Initializing parameters")
    params = {
        'neural_data_bin_size': 0.01, # 10 ms in seconds
        'smooth_spike_counts': True,
        'is_grace': False,
        'num_cpus': min(8, cpu_count())
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    params = curate_data.add_raw_data_dir_to_params(params)
    params = curate_data.add_paths_to_all_data_files_to_params(params)
    params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(params)
    logger.info("Parameters initialized successfully")
    return params



def _plot_mean_behavioral_spiking_response(
    eye_mvm_behav_df, spike_times_df, sparse_nan_removed_sync_gaze_df, params
):
    """
    Generate plots of mean spiking activity for each neuron around saccade and fixation events.
    Separate subplots are created for agents m1 and m2 in the same plot.
    Rows correspond to "to" saccades, "from" saccades, and "fixation on".
    Parameters:
        eye_mvm_behav_df (pd.DataFrame): Behavioral dataframe with saccade/fixation start-stop indices and labels.
        spike_times_df (pd.DataFrame): Dataframe containing spiking data for units in each session.
        sparse_nan_removed_sync_gaze_df (pd.DataFrame): Gaze dataframe with neural timeline and positions.
        params (dict): Configuration parameters including root directory and bin size.
    Returns:
        None
    """
    logger.info("Generating mean activity response plots")
    today_date = datetime.now().strftime("%Y-%m-%d")
    root_dir = os.path.join(
        params['root_data_dir'], "plots", "mean_behavioral_spiking_response", today_date
    )
    os.makedirs(root_dir, exist_ok=True)
    bin_size = params.get("neural_data_bin_size", 0.01)
    time_window = 1
    roi_labels = ["face", "mouth", "eyes_nf", "object", "out_of_roi"]
    session_tasks = []
    for session_name, session_behav_df in eye_mvm_behav_df.groupby("session_name"):
        session_spike_df = spike_times_df[
            spike_times_df["session_name"] == session_name
        ]
        session_gaze_df = sparse_nan_removed_sync_gaze_df[
            sparse_nan_removed_sync_gaze_df["session_name"] == session_name
        ]
        session_dir = os.path.join(root_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        session_tasks.append((session_name, session_behav_df, session_spike_df, session_gaze_df, session_dir, bin_size, time_window, roi_labels, params))
    logger.info(f"Running parallel processing with {params['num_cpus']} CPUs")
    with Pool(processes=params['num_cpus']) as pool:
        list(tqdm(pool.imap(__make_plots_for_session, session_tasks), total=len(session_tasks), desc="Plotting for sessions"))

def __make_plots_for_session(task):
    """
    Process a single session to generate plots for all units.
    Parameters:
        task (tuple): Contains session-specific data and parameters.
    Returns:
        None
    """
    session_name, session_behav_df, session_spike_df, session_gaze_df, session_dir, bin_size, time_window, roi_labels, params = task
    for _, unit in tqdm(session_spike_df.iterrows(), total=len(session_spike_df), desc=f"Processing units in {session_name}"):
        __plot_unit_activity(
            unit, session_behav_df, session_gaze_df, bin_size, time_window, session_dir, roi_labels, params
        )

def __plot_unit_activity(
    unit, session_behav_df, session_gaze_df, bin_size, time_window, session_dir, roi_labels, params
):
    """
    Generate a plot for a single unit showing mean spiking activity around saccade and fixation events.
    Separate subplots are created for agents m1 and m2 and for saccades "to", "from", and "fixation on" ROIs.
    Parameters:
        unit (pd.Series): Row from spike_times_df representing a single unit.
        session_behav_df (pd.DataFrame): Behavioral data for the session.
        session_gaze_df (pd.DataFrame): Gaze data for the session.
        bin_size (float): Size of bins for spiking activity (in seconds).
        time_window (float): Time window around saccade/fixation start (in seconds).
        session_dir (str): Directory to save the plot.
        roi_labels (list): List of regions of interest to consider.
        params (dict): Configuration parameters including smoothing option.
    Returns:
        None
    """
    unit_region = unit["region"]
    unit_uuid = unit["unit_uuid"]
    spike_times = np.array(unit["spike_ts"])
    fig, axs = plt.subplots(3, 2, figsize=(12, 18), sharex=True, sharey=True)
    for row_idx, event_type in enumerate(["saccade_to", "saccade_from", "fixation_on"]):
        for agent_idx, agent in enumerate(["m1", "m2"]):
            ___plot_mean_activity_for_event_and_agent(
                event_type, roi_labels, row_idx, agent_idx, unit, session_behav_df, session_gaze_df,
                spike_times, bin_size, time_window, axs, params
            )
    plt.suptitle(f"Mean Activity for UUID: {unit_uuid}, Region: {unit_region}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(session_dir, f"region_{unit_region}_unit_{unit_uuid}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

def ___plot_mean_activity_for_event_and_agent(
    event_type, roi_labels, row_idx, agent_idx, unit, session_behav_df, session_gaze_df,
    spike_times, bin_size, time_window, axs, params
):
    """
    Plot mean activity for a specific event type and agent, including all ROIs in the same subplot.
    Parameters:
        event_type (str): Type of event ('saccade_to', 'saccade_from', 'fixation_on').
        roi_labels (list): List of ROIs to include.
        row_idx (int): Index for the event type.
        agent_idx (int): Index for the agent.
        unit (pd.Series): Unit data.
        session_behav_df (pd.DataFrame): Behavioral data for the session.
        session_gaze_df (pd.DataFrame): Gaze data for the session.
        spike_times (np.ndarray): Spike timestamps.
        bin_size (float): Bin size for spiking activity.
        time_window (float): Time window around event start (in seconds).
        axs (np.ndarray): Axes for plotting.
        params (dict): Configuration parameters including smoothing option.
    Returns:
        None
    """
    ax = axs[row_idx, agent_idx]
    agent = ["m1", "m2"][agent_idx]
    agent_behav_df = session_behav_df[session_behav_df["agent"] == agent]
    agent_gaze_df = session_gaze_df[session_gaze_df["agent"] == agent]
    for roi in roi_labels:
        mean_activity, timeline = ____compute_mean_activity(
            event_type, roi, agent_behav_df, agent_gaze_df, spike_times, bin_size, time_window, params
        )
        if mean_activity is not None:
            ax.plot(timeline[:-1], mean_activity, label=roi)
    ax.set_title(f"{agent.upper()} - {event_type.replace('_', ' ').capitalize()}")
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Firing Rate (Hz)")

def ____compute_mean_activity(
    event_type, roi, agent_behav_df, agent_gaze_df, spike_times, bin_size, time_window, params
):
    """
    Compute mean spiking activity around specific events (saccades or fixations) for a given ROI.
    Parameters:
        event_type (str): Type of event ('saccade_to', 'saccade_from', 'fixation_on').
        roi (str): Region of interest to filter events.
        agent_behav_df (pd.DataFrame): Behavioral data for the agent.
        agent_gaze_df (pd.DataFrame): Gaze data for the agent.
        spike_times (np.ndarray): Spike timestamps.
        bin_size (float): Bin size for spiking activity.
        time_window (float): Time window around event start (in seconds).
        params (dict): Configuration parameters including smoothing option.
    Returns:
        mean_activity (np.ndarray): Mean spiking activity.
        timeline (np.ndarray): Timeline corresponding to the mean activity.
    """
    if event_type.startswith("saccade"):
        direction = event_type.split("_")[1]
        event_column = f"saccade_{direction}"
    else:
        event_column = "fixation_location"
    mean_activity = []
    timeline = np.arange(-time_window, time_window + bin_size, bin_size)
    for _, run_row in agent_behav_df.iterrows():
        if event_type.startswith("saccade"):
            events = run_row[event_column]
            start_stops = run_row["saccade_start_stop"]
        else:
            events = run_row[event_column]
            start_stops = run_row["fixation_start_stop"]
        run_number = run_row["run_number"]
        relevant_events = [
            event_idx for event_idx, labels in enumerate(events)
            if roi in labels or (roi == "object" and any(l in labels for l in ["left_nonsocial_object", "right_nonsocial_object"]))
        ]
        if not relevant_events:
            continue
        run_gaze_data = agent_gaze_df[agent_gaze_df["run_number"] == run_number]
        if run_gaze_data.empty:
            continue
        neural_timeline = run_gaze_data.iloc[0]["neural_timeline"]
        for event_idx in relevant_events:
            event_start = start_stops[event_idx][0]
            if event_start >= len(neural_timeline):
                continue
            event_time = neural_timeline[event_start]
            bins = np.linspace(
                event_time - time_window,
                event_time + time_window,
                int(2 * time_window / bin_size) + 1
            ).flatten()
            spike_counts, _ = np.histogram(spike_times, bins=bins)
            firing_rate = spike_counts / bin_size  # Convert spike counts to firing rate
            if params['smooth_spike_counts']:
                firing_rate = np.convolve(firing_rate, np.ones(5)/5, mode='same')
            mean_activity.append(firing_rate)
    if len(mean_activity) > 0:
        mean_activity = np.mean(mean_activity, axis=0)
        return mean_activity, timeline
    return None, None



if __name__ == "__main__":
    main()
