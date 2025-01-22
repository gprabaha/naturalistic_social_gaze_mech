import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, cpu_count

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


if __name__ == "__main__":
    main()



'''
This is the rough code to implement eh NeuroXidence algorithm
Will have to check things out and then implement it in our data
'''


import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def preprocess_spike_trains(spike_times, units, bin_size=1, duration=None):
    """
    Converts spike times into a binary matrix where rows are neurons and columns are time bins.
    """
    if duration is None:
        duration = max([max(spike_times[u]) if len(spike_times[u]) > 0 else 0 for u in units])
    
    num_bins = int(duration / bin_size)
    spike_matrix = np.zeros((len(units), num_bins), dtype=int)
    
    unit_to_idx = {unit: i for i, unit in enumerate(units)}
    for unit, spikes in zip(units, spike_times):
        bin_indices = (np.array(spikes) / bin_size).astype(int)
        spike_matrix[unit_to_idx[unit], bin_indices] = 1
    
    return spike_matrix

def detect_joint_spike_events(spike_matrix, tau_c=5):
    """
    Detects joint-spike events (JSEs) where multiple neurons fire within a given window.
    """
    num_neurons, num_bins = spike_matrix.shape
    joint_spike_events = []
    
    for t in range(num_bins - tau_c):
        active_neurons = np.where(np.sum(spike_matrix[:, t:t+tau_c], axis=1) > 0)[0]
        if len(active_neurons) > 1:
            joint_spike_events.append(set(active_neurons))
    
    return joint_spike_events

def generate_surrogate_data(spike_matrix, tau_r=20, num_surrogates=20):
    """
    Generates surrogate datasets by jittering spike trains within a larger window.
    """
    num_neurons, num_bins = spike_matrix.shape
    surrogate_data = []
    
    for _ in range(num_surrogates):
        jittered_spike_matrix = np.zeros_like(spike_matrix)
        for neuron in range(num_neurons):
            spikes = np.where(spike_matrix[neuron] == 1)[0]
            jittered_spikes = spikes + np.random.randint(-tau_r, tau_r, size=len(spikes))
            jittered_spikes = jittered_spikes[(jittered_spikes >= 0) & (jittered_spikes < num_bins)]
            jittered_spike_matrix[neuron, jittered_spikes] = 1
        surrogate_data.append(jittered_spike_matrix)
    
    return surrogate_data

def compute_significance(real_jse, surrogate_jses):
    """
    Compares real joint-spike events to surrogate data and computes statistical significance.
    """
    real_counts = [len(jse) for jse in real_jse]
    surrogate_counts = [[len(detect_joint_spike_events(surr)) for surr in surrogate_jses]]
    
    p_values = []
    for real, surrogate in zip(real_counts, zip(*surrogate_counts)):
        p = wilcoxon(real, surrogate, alternative='greater').pvalue
        p_values.append(p)
    
    return p_values

def detect_synchrony(eye_mvm_behav_df, sparse_nan_removed_sync_gaze_df, spike_times_df, session_name, tau_c=5, tau_r=20, num_surrogates=20):
    """
    Detects neuronal synchrony using NeuroXidence for fixations.
    """
    session_data = eye_mvm_behav_df[eye_mvm_behav_df["session_name"] == session_name]
    fixation_events = session_data[(session_data["agent"] == "m1") & 
                                   (session_data["fixation_location"].isin(["face", "object", "out_of_roi"]))]
    fixation_times = fixation_events[["run_number", "fixation_location", "start_stop"]]
    
    neural_fixation_times = []
    for _, row in fixation_times.iterrows():
        run_num = row["run_number"]
        start, stop = row["start_stop"]
        neural_timeline = sparse_nan_removed_sync_gaze_df[
            (sparse_nan_removed_sync_gaze_df["session_name"] == session_name) &
            (sparse_nan_removed_sync_gaze_df["run_number"] == run_num)
        ]["neural_timeline"].values[0]
        
        neural_start = neural_timeline[start] if start < len(neural_timeline) else None
        neural_stop = neural_timeline[stop] if stop < len(neural_timeline) else None
        
        if neural_start is not None and neural_stop is not None:
            neural_fixation_times.append((run_num, row["fixation_location"], neural_start, neural_stop))
    
    neural_fixation_df = pd.DataFrame(neural_fixation_times, columns=["run_number", "fixation_location", "neural_start", "neural_stop"])
    
    spike_events = []
    for _, row in neural_fixation_df.iterrows():
        neural_start, neural_stop = row["neural_start"], row["neural_stop"]
        session_spikes = spike_times_df[spike_times_df["session_name"] == session_name]
        spiking_events = session_spikes[(session_spikes["spike_time"] >= neural_start) & 
                                        (session_spikes["spike_time"] <= neural_stop)]
        if not spiking_events.empty:
            spike_events.append(spiking_events)
    
    if not spike_events:
        return pd.DataFrame()
    
    spike_times_list = [spiking_events["spike_time"].values for spiking_events in spike_events]
    spike_units_list = [spiking_events["unit"].values for spiking_events in spike_events]
    
    spike_matrix = preprocess_spike_trains(spike_times_list, spike_units_list)
    real_jse = detect_joint_spike_events(spike_matrix, tau_c)
    surrogate_jses = generate_surrogate_data(spike_matrix, tau_r, num_surrogates)
    significance = compute_significance(real_jse, surrogate_jses)
    
    return pd.DataFrame(zip(real_jse, significance), columns=["pattern", "p_value"])