import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import sys
from pathlib import Path
# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from data_manager import DataManager

import pdb

def main():

    params = {}
    params.update({
        'sampling_rate': 1 / 1000,

        'is_cluster': True,
        'is_grace': False,
        'use_parallel': True,
        'submit_separate_jobs_for_sessions': True,
        'use_toy_data': False,
        'remake_toy_data': False,
        'num_cpus': None,


        'remake_gaze_data_df': False,
        'remake_spike_times_df': False,
        'remake_nan_removed_gaze_data_df': False,

        'remake_fix_and_sacc': False,
        'try_using_single_run': False,
        'recompute_fix_and_saccades_through_hpc_jobs': True,
        'hpc_job_output_subfolder': 'single_run_fix_saccade_results',
        'remake_binary_timeseries': False,
        'remake_scaled_autocorr': False,

        'remake_neural_timeseries': False,
        'neural_data_bin_size': 0.001,  #1ms = 0.001s
        'downsample_bin_size': 0.01,
        'smoothing_sigma': 10
    })

    # import and organize data
    data_manager = DataManager(params)
    data_manager.run()
    firing_rate_df = data_manager.neural_fr_timeseries_df
    pdb.set_trace()
    return 0


if __name__ == "__main__":
    main()