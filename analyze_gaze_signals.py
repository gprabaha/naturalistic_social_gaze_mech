#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:19:31 2024

@author: pg496
"""

import logging

from data_manager import DataManager

def main():
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        format='%(name)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
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
    data_manager = DataManager(params)
    data_manager.run()


if __name__ == "__main__":
    main()


