import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from data_manager import DataManager

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
        'do_local_reclustering_in_parallel': False,
        'remake_gaze_data_df': False,
        'remake_nan_removed_gaze_data_df': False,
        'remake_fix_and_sacc': False,
        'try_using_single_run': False,
        'recompute_fix_and_saccades_through_hpc_jobs': True,
        'hpc_job_output_subfolder': 'single_run_fix_saccade_results',
        'remake_binary_timeseries': False,
        'remake_scaled_autocorr': False
    })

    # import and organize data
    data_manager = DataManager(params)


if __name__ == "__main__":
    main()