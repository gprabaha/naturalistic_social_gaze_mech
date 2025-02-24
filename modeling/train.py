import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import train_config

import sys
import os
from pathlib import Path
# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import curate_data
import load_data
import itertools
from dataset import FiringRateDataset
import pdb

def _initialize_params(
    remake_firing_rate_df=False,
    neural_data_bin_size=10,
    smooth_spike_counts=True,
    guassian_smoothing_sigma=2,
    time_window_before_event=500,
    is_cluster=False,
    path_name=None
):
    params = {
        'remake_firing_rate_df': remake_firing_rate_df,
        'neural_data_bin_size': neural_data_bin_size,  # 10 ms in seconds
        'smooth_spike_counts': smooth_spike_counts,
        'gaussian_smoothing_sigma': guassian_smoothing_sigma,
        'time_window_before_event': time_window_before_event,
        'is_cluster': is_cluster,
        'path_name': path_name
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    return params

def main():

    ### PARAMETERS ###
    parser = train_config.config_parser()
    args = parser.parse_args()

    # Load processed dataframe
    params = _initialize_params(
        path_name="/Users/John/naturalistic_social_gaze_mech/social_gaze"
    )
    behav_firing_rate_df_file_path = os.path.join(
        params['processed_data_dir'], 'behavioral_firing_rate_df.pkl'
    )
    print('loading data...')
    df = load_data.get_data_df(behav_firing_rate_df_file_path)

    print('creating fr dataset...')
    dataset = FiringRateDataset(df)
    print('creating dataloader...')
    dataloader = DataLoader(dataset)

    for batch in dataloader:
        print(batch[0].shape)

if __name__ == "__main__":
    main()