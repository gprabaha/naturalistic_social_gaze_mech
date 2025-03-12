import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

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
        'neural_data_bin_size': 0.01,  # 10 ms in seconds
        'smooth_spike_counts': True,
        'time_window_before_and_after_event_for_psth': 0.5,
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
    fix_binary_vector_file = os.path.join(
        processed_data_dir, 'fix_binary_vector_df.pkl'
    )
    logger.info("Loading data files")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_df = load_data.get_data_df(spike_times_file_path)
    fix_binary_vector_df = load_data.get_data_df(fix_binary_vector_file)

    pdb.set_trace()


    logger.info("Script finished running!")




if __name__ == "__main__":
    main()

