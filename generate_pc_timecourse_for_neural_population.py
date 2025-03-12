import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import gc


from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
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
        'recompute_firing_rate': False,
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
    logger.info("Loading data files")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_df = load_data.get_data_df(spike_times_file_path)

    # Perform the merge
    eye_mvm_behav_df_with_neural_timeline = eye_mvm_behav_df.merge(
        sparse_nan_removed_sync_gaze_df[['session_name', 'interaction_type', 'run_number', 'agent', 'positions', 'neural_timeline']],
        on=['session_name', 'interaction_type', 'run_number', 'agent'],
        how='left'  # Use 'left' join to preserve all rows in eye_mvm_behav_df
    )
    del sparse_nan_removed_sync_gaze_df, eye_mvm_behav_df
    gc.collect()

    # Define the file path for storing the computed firing rate DataFrame
    firing_rate_file = os.path.join(params['processed_data_dir'], 'firing_rate_df.pkl')

    # Check if we need to recompute or load from file
    if params.get('recompute_firing_rate', False) or not os.path.exists(firing_rate_file):
        fixation_firing_rate_df = compute_firing_rate_matrix(eye_mvm_behav_df_with_neural_timeline, spike_times_df, params)
        fixation_firing_rate_df.to_pickle(firing_rate_file)  # Save to file
        logger.info(f"Firing rate df saved to: {firing_rate_file}")
    else:
        logger.info(f"Loading existing firing rate df from: {firing_rate_file}")
        fixation_firing_rate_df = load_data.get_data_df(firing_rate_file)  # Load from file

    project_and_plot_pcs(fixation_firing_rate_df, params)

    logger.info("Script finished running!")



##############################
# ** FUNCTION DEFINITIONS ** #
##############################


def compute_firing_rate_matrix(eye_mvm_behav_df, spike_times_df, params):
    """
    Computes the trial-by-trial, binned firing rate timecourse matrix for each unit in each session and region.
    Parameters:
    - eye_mvm_behav_df: DataFrame containing fixation events with neural timeline and positions.
    - spike_times_df: DataFrame containing spike timestamps for each unit.
    - params: Dictionary containing:
        - 'neural_data_bin_size': Bin size for firing rate calculation (seconds).
        - 'gaussian_smoothing_sigma': Standard deviation for Gaussian smoothing.
        - 'time_window_before_event_for_psth': Time before fixation start (seconds).
        - 'time_window_after_event_for_psth': Time after fixation start (seconds).
    Returns:
    - firing_rate_df: DataFrame with columns ['session_name', 'region', 'unit_uuid', 'firing_rate_matrix']
    """
    logger.info("Computing trial-wise firing rate matrix for all units in all sessions")
    bin_size = params.get('neural_data_bin_size', 0.01)
    smooth_sigma = params.get('gaussian_smoothing_sigma', 2)
    time_window_before = params.get('time_window_before_event_for_psth', 0.5)  # 0.5 seconds before
    time_window_after = params.get('time_window_after_event_for_psth', 1.0)   # 1.0 second after
    results = []

    # Step 1: Apply fixation type classification to all rows in `eye_mvm_behav_df`
    def get_fixation_type(fix_locs):
        """
        Returns a LIST of all matching fixation types instead of a single label.
        """
        fix_type = []
        for sublist in fix_locs:
            appended = 0
            for loc in sublist:
                if 'face' in loc or 'mouth' in loc:
                    fix_type.append('face')
                    appended = 1
                    break  # Prioritize the first match per sublist
                if 'object' in loc:
                    fix_type.append('object')
                    appended = 1
                    break
            if appended == 0:
                fix_type.append('out_of_roi')
        return fix_type  # Return list

    # Apply fixation type function to the entire DataFrame
    eye_mvm_behav_df['fixation_type'] = eye_mvm_behav_df['fixation_location'].apply(get_fixation_type)

    # Step 2: Iterate over sessions
    for session, session_spike_df in tqdm(spike_times_df.groupby('session_name'), desc='FR calc for sessions'):
        # Filter fixation data for this session and agent 'm1'
        session_m1_fixations = eye_mvm_behav_df[
            (eye_mvm_behav_df['session_name'] == session) & 
            (eye_mvm_behav_df['agent'] == 'm1')
        ]

        # Step 3: Iterate over units in this session
        for _, unit_row in session_spike_df.iterrows():
            unit_uuid = unit_row['unit_uuid']
            region = unit_row['region']
            spike_times = np.array(unit_row['spike_ts'])  # Convert spike times to NumPy array
            per_trial_frs = {'face': [], 'object': []}  # Separate lists for fixation types

            # Step 4: Iterate over runs in this session
            for run, run_fixations in session_m1_fixations.groupby('run_number'):
                
                # Step 5: Iterate through each fixation row in this run
                for _, fixation_row in run_fixations.iterrows():
                    fixation_types = fixation_row['fixation_type']  # List of fixation types in this row
                    fixation_start_stop = fixation_row['fixation_start_stop']  # List of start/stop indices
                    neural_times = np.array(fixation_row['neural_timeline']).flatten()

                    # Step 6: Iterate through each fixation type in this row
                    for fix_idx, fixation_type in enumerate(fixation_types):
                        if fixation_type in ['face', 'object']:  # Process only relevant fixation types
                            fixation_start_index = fixation_start_stop[fix_idx][0]  # Get start index
                            fixation_start_time = neural_times[fixation_start_index]  # Get corresponding neural time
                            # Define time window
                            start_time = fixation_start_time - time_window_before
                            end_time = fixation_start_time + time_window_after
                            # Ensure consistent number of bins
                            n_bins = int(round((end_time - start_time) / bin_size))
                            bins = np.linspace(start_time, end_time, n_bins + 1)  # +1 to include last bin edge
                            # Bin spike counts
                            binned_spike_counts, _ = np.histogram(spike_times, bins=bins)
                            # Gaussian smoothing
                            smoothed_firing_rate = gaussian_filter1d(binned_spike_counts / bin_size, sigma=smooth_sigma)
                            # Store the firing rate in the appropriate fixation type list
                            per_trial_frs[fixation_type].append(smoothed_firing_rate)

            # Convert lists to numpy arrays outside the loop after processing all runs
            for fixation_type in ['face', 'object']:
                per_trial_frs[fixation_type] = np.array(per_trial_frs[fixation_type]) if per_trial_frs[fixation_type] else np.empty((0,))
                # Store results
                results.append({
                    'session_name': session,
                    'region': region,
                    'unit_uuid': unit_uuid,
                    'fixation_type': fixation_type,
                    'firing_rate_matrix': per_trial_frs[fixation_type]  # Now contains numpy arrays
                })

    # Step 7: Convert to DataFrame (Removing run_number)
    firing_rate_df = pd.DataFrame(results)
    logger.info("Trial-wise FR calculation finished successfully")
    return firing_rate_df



def project_and_plot_pcs(fixation_firing_rate_df, params):
    """Projects face and object fixation-related firing rates to PCA space and plots trajectories.

    Args:
        fixation_firing_rate_df (pd.DataFrame): DataFrame containing firing rate matrices per unit.
        params (dict): Dictionary of parameters including `root_data_dir`.
    """
    # Set up export directory
    today_date = datetime.today().strftime('%Y-%m-%d')
    export_dir = os.path.join(params['root_data_dir'], "plots", "neural_fr_pc_traces", today_date)
    os.makedirs(export_dir, exist_ok=True)
    unique_regions = fixation_firing_rate_df['region'].unique()
    num_regions = len(unique_regions)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': '3d'})
    fig.suptitle("PC Trajectories of Face & Object Fixations", fontsize=14)

    for idx, region in enumerate(unique_regions):
        if idx >= 4:  # Limit to 2x2 grid (4 regions)
            break
        
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        logger.info(f"Processing region: {region}")
        # Extract firing rate matrices for face and object fixations
        regional_df = fixation_firing_rate_df[fixation_firing_rate_df['region'] == region]
        firing_rates = {}

        for fix_type in ['face', 'object']:
            fix_type_df = regional_df[regional_df['fixation_type'] == fix_type]
            if fix_type_df.empty:
                firing_rates[fix_type] = None  # Mark as missing
            else:
                # Extract firing rate matrices (mean over axis 0 for each unit)
                unit_firing_rates = np.array([np.mean(mat, axis=0) for mat in fix_type_df['firing_rate_matrix']])
                firing_rates[fix_type] = unit_firing_rates  # Shape: (num_units, num_timepoints)

        face_data = firing_rates['face']
        object_data = firing_rates['object']
        
        # Stack units across conditions along the **neural axis** (not time!)
        combined_firing_rates = np.vstack([face_data, object_data])  # Shape: (2*num_units, num_timepoints)

        # **Fit PCA on the neural dimension**, keeping time intact
        pca = PCA(n_components=3)
        pca.fit(combined_firing_rates.T)  # Transpose to shape (num_timepoints, 2*num_units) before fitting

        # **Zero-Pad Face & Object Before Transforming**
        face_padded = np.vstack([face_data, np.zeros_like(face_data)])  # Shape: (2*num_units, num_timepoints)
        object_padded = np.vstack([np.zeros_like(object_data), object_data])  # Shape: (2*num_units, num_timepoints)
        
        # **Transform padded versions**
        projected_face = pca.transform(face_padded.T)  # Shape: (num_timepoints, 3)
        projected_object = pca.transform(object_padded.T)  # Shape: (num_timepoints, 3)

        # Plot PC trajectories for face and object fixations in 3D
        ax.plot(projected_face[:, 0], projected_face[:, 1], projected_face[:, 2], label="Face", color="blue")
        ax.plot(projected_object[:, 0], projected_object[:, 1], projected_object[:, 2], label="Object", color="red")
        ax.set_title(f"Region: {region}", fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(export_dir, "pc_trajectories.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved PCA trajectories plot to: {plot_path}")







if __name__ == "__main__":
    main()

