#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14::51:42 2024

@author: pg496
"""

import logging
import os
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np

import util
import curate_data
import load_data
import fix_and_saccades
import analyze_data
import plotter

import pdb

import load_data
from hpc_shuffled_cross_corr import HPCShuffledCrossCorr


import os
import logging
import multiprocessing
import pandas as pd
from joblib import Parallel, delayed
import curate_data
import load_data
import analyze_data
import fix_and_saccades
import plotter

'''
To implement:

- For each run make sure that the behavioral dataframe is updated such that if the `from` field is specified
then the `to` field is always `anywhere`. Similarly, if the `to` field is specified then the same for `from`

- Update the `elsewhere` lavelled behavior to `out_of_roi` to make it less confusing and distinguishing from `anywhere`

- Make a dataframe where for each run, we have the entire behavioral matrix where the rows represent the 
binary timeseries of a particular behavior. Assign a column with the behavioral `from-to` labels as well
    - Now fit the SLDS model to this
    - We need to create a similar behavioral matrix but where the ROIs are shiuffled or something and then show
    that the prediction is off in that case

'''

class DataManager:
    def __init__(self, params):
        """Initialize the DataManager with parameters, logger setup, and CPU detection."""
        self._setup_logger()
        self.params = params
        self._find_n_cores()
        self._initialize_class_objects()


    def _setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)


    def _find_n_cores(self):
        """Determine the number of CPU cores available, prioritizing SLURM if available."""
        try:
            slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
            num_cpus = int(slurm_cpus)
            self.logger.info(f"SLURM detected {num_cpus} CPUs")
        except Exception as e:
            self.logger.warning(f"Failed to detect cores with SLURM_CPUS_ON_NODE: {e}")
            num_cpus = multiprocessing.cpu_count()
            self.logger.info(f"Multiprocessing detected {num_cpus} CPUs")
        os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus)
        self.num_cpus = num_cpus
        self.params['num_cpus'] = num_cpus


    def _initialize_class_objects(self):
        """Initialize class object attributes to None."""
        self.recording_sessions_and_monkeys = None
        self.gaze_data_df = None
        self.spike_times_df = None
        self.missing_data_paths = None
        self.synchronized_gaze_data_df = None
        self.fixation_df = None
        self.saccade_df = None
        self.binary_behav_timeseries_df = None
        self.binary_timeseries_scaled_autocorr_df = None
        self.neural_fr_timeseries_df = None


    def populate_params_with_data_paths(self):
        """Populate the params dictionary with various data paths needed for processing."""
        self.params = curate_data.add_root_data_to_params(self.params)
        self.params = curate_data.add_processed_data_to_params(self.params)
        self.params = curate_data.add_raw_data_dir_to_params(self.params)
        self.params = curate_data.add_paths_to_all_data_files_to_params(self.params)
        self.params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(self.params)


    def get_data(self):
        """Load or compute necessary gaze and spike data, filtering for sessions with ephys data."""
        self.recording_sessions_and_monkeys = self._load_ephys_sessions()
        # self.gaze_data_df, self.missing_data_paths = self._load_or_compute_gaze_data()
        # self.gaze_data_df = self._filter_sessions_with_ephys(self.gaze_data_df)
        self.spike_times_df = self._load_or_compute_spike_times()
        self.spike_times_df = self._filter_sessions_with_ephys(self.spike_times_df)


    def _load_ephys_sessions(self):
        """Load the list of sessions with ephys data."""
        path_to_ephys_sessions = os.path.join(self.params['processed_data_dir'], 'ephys_days_and_monkeys.pkl')
        return load_data.load_recording_days(path_to_ephys_sessions)


    def _load_or_compute_gaze_data(self):
        """Load or compute gaze data and missing paths DataFrames."""
        gaze_data_file_path = os.path.join(self.params['processed_data_dir'], 'gaze_data_df.pkl')
        missing_data_paths_file_path = os.path.join(self.params['processed_data_dir'], 'missing_data_paths.pkl')
        if self.params.get('remake_gaze_data_df', False) or not os.path.exists(gaze_data_file_path):
            self.logger.info("Generating gaze data and missing paths.")
            gaze_data, missing_paths = curate_data.make_gaze_data_df(self.params)
            gaze_data.to_pickle(gaze_data_file_path)
            missing_paths.to_pickle(missing_data_paths_file_path)
            return gaze_data, missing_paths
        else:
            self.logger.info("Loading existing gaze data and missing paths.")
            return load_data.get_gaze_data_df(gaze_data_file_path, missing_data_paths_file_path)


    def _filter_sessions_with_ephys(self, df):
        """Filter the DataFrame to include only sessions that have electrophysiology data."""
        if self.recording_sessions_and_monkeys is None:
            self.logger.error("recording_sessions_and_monkeys is not loaded. Ensure get_data() is called first.")
            return pd.DataFrame()  # Return empty DataFrame if the sessions are not loaded
        filtered_df = df[df['session_name'].isin(self.recording_sessions_and_monkeys['session_name'])]
        self.logger.info(f"Filtered to {len(filtered_df)} rows with electrophysiology data.")
        return filtered_df


    def _load_or_compute_spike_times(self):
        """Load or compute spike times DataFrame."""
        spike_times_file_path = os.path.join(self.params['processed_data_dir'], 'spike_times_df.pkl')
        if self.params.get('remake_spike_times_df', False) or not os.path.exists(spike_times_file_path):
            self.logger.info("Generating spike times data.")
            spike_times = curate_data.make_spike_times_df(self.params)
            spike_times.to_pickle(spike_times_file_path)
            return spike_times
        else:
            self.logger.info("Loading existing spike times data.")
            return load_data.get_spike_times_df(spike_times_file_path)


    def prune_data(self):
        """Prune NaN values from gaze data."""
        synchronized_gaze_data_file_path = os.path.join(self.params['processed_data_dir'], 'synchronized_gaze_data_df.pkl')
        if self.params.get('remake_synchronized_gaze_data_df', False) or not os.path.exists(synchronized_gaze_data_file_path):
            self.logger.info("Pruning NaN values in the beginning and end of gaze data.")
            synchronized_gaze_data = curate_data.synchronize_m1_and_m2_timelines_by_pruning_flanking_nans(self.gaze_data_df)
            synchronized_gaze_data.to_pickle(synchronized_gaze_data_file_path)
            self.synchronized_gaze_data_df = synchronized_gaze_data
            self.gaze_data_df = None
        else:
            self.logger.info("Loading pruned gaze data.")
            self.synchronized_gaze_data_df = load_data.get_synchronized_gaze_data_df(synchronized_gaze_data_file_path)
            self.gaze_data_df = None


    def analyze_behavior(self):
        """Analyze behavior by detecting fixations and saccades and computing binary timeseries and autocorrelations."""
        self.fixation_df, self.saccade_df = self._load_or_compute_fixations_and_saccades()
        pdb.set_trace()
        self.binary_behav_timeseries_df = self._load_or_compute_binary_behav_timeseries()

        
        # self.crosscorrelation_df_between_all_m1_amd_m2_behavior = self._load_or_compute_crosscorr_df()
        # self.shuffled_crosscorrelation_df_between_all_m1_amd_m2_behavior = analyze_data.compute_behavioral_cross_correlations(
        #     self.binary_behav_timeseries_df, self.params, shuffled=True, serial=False, remake_results=True)
        # shuffled_crosscorr_file_path = os.path.join(self.params['processed_data_dir'], 'inter_agent_shuffled_crosscorrelation_df.pkl')
        # self.shuffled_crosscorrelation_df_between_all_m1_amd_m2_behavior.to_pickle(shuffled_crosscorr_file_path)
        # self.logger.info(f"Saved interagent cross-correlation df to: {shuffled_crosscorr_file_path}")
        # self.create_and_submit_shuffled_cross_corr_jobs()
        # self.binary_timeseries_scaled_auto_and_crosscorr_df = self._load_or_compute_binary_timeseries_auto_and_crosscorr()
        # self.neural_fr_timeseries_df = self._load_or_compute_neural_fr_timeseries_df()


    def create_and_submit_shuffled_cross_corr_jobs(self):
        binary_timeseries_file_path = os.path.join(self.params['processed_data_dir'], 'binary_behav_timeseries.pkl')
        self.logger.info(f'Loading binary behavioral df from {binary_timeseries_file_path}')
        self.binary_behav_timeseries_df = load_data.load_binary_timeseries_df(binary_timeseries_file_path)
        grouped_keys = self.binary_behav_timeseries_df.groupby(['session_name', 'interaction_type', 'run_number']).groups.keys()
        hpc_handler = HPCShuffledCrossCorr(self.params)
        num_cpus_per_job = 18
        num_shuffles = 100
        job_file_path = hpc_handler.generate_job_file(grouped_keys, binary_timeseries_file_path, num_cpus_per_job, num_shuffles)
        hpc_handler.submit_job_array(job_file_path, num_cpus_per_job)


    def _load_or_compute_fixations_and_saccades(self):
        """Load or compute fixation and saccade DataFrames."""
        fixation_file_path = os.path.join(self.params['processed_data_dir'], 'fixation_df.pkl')
        saccade_file_path = os.path.join(self.params['processed_data_dir'], 'saccade_df.pkl')
        if self.params.get('remake_fix_and_sacc', False) or not (os.path.exists(fixation_file_path) and os.path.exists(saccade_file_path)):
            self.logger.info("Detecting fixations and saccades.")
            fixation_df, saccade_df = fix_and_saccades.detect_fixations_and_saccades(self.synchronized_gaze_data_df, self.params)
            # fixation_df, saccade_df = load_data.load_fixation_and_saccade_dfs(fixation_file_path, saccade_file_path)
            # use_parallel = self.params['use_parallel']
            use_parallel = False
            fixation_df = curate_data.add_fixation_rois_in_dataframe(fixation_df, self.synchronized_gaze_data_df, self.num_cpus, use_parallel)
            saccade_df = curate_data.add_saccade_rois_in_dataframe(saccade_df, self.synchronized_gaze_data_df, self.num_cpus, use_parallel)
            fixation_df.to_pickle(fixation_file_path)
            saccade_df.to_pickle(saccade_file_path)
            return fixation_df, saccade_df
        else:
            self.logger.info("Loading existing fixation and saccade data.")
            return load_data.load_fixation_and_saccade_dfs(fixation_file_path, saccade_file_path)


    def _load_or_compute_binary_behav_timeseries(self):
        """Load or compute binary behavior timeseries DataFrame."""
        binary_timeseries_file_path = os.path.join(self.params['processed_data_dir'], 'binary_behav_timeseries.pkl')
        if self.params.get('remake_binary_timeseries', False) or not os.path.exists(binary_timeseries_file_path):
            self.logger.info("Generating binary behavior timeseries.")
            # use_parallel = self.params['use_parallel']
            use_parallel = False
            fixation_timeline_df = analyze_data.create_binary_timeline_for_behavior(
                self.fixation_df,
                self.synchronized_gaze_data_df,
                self.num_cpus,
                behavior_type='fixation',
                use_parallel=use_parallel)
            saccade_timeline_df = analyze_data.create_binary_timeline_for_behavior(
                self.saccade_df,
                self.synchronized_gaze_data_df,
                self.num_cpus,
                behavior_type='saccade',
                use_parallel=use_parallel)
            # Assuming fixation_timeline_df and saccade_timeline_df are already created
            binary_behav_timeseries_df = pd.concat([fixation_timeline_df, saccade_timeline_df], ignore_index=True)
            binary_behav_timeseries_df = analyze_data.merge_left_and_right_object_timelines_in_behav_df(binary_behav_timeseries_df)
            binary_behav_timeseries_df.to_pickle(binary_timeseries_file_path)
            self.logger.info(f"Saved binary behavior timeseries to: {binary_timeseries_file_path}")
            return binary_behav_timeseries_df
        else:
            self.logger.info("Loading existing binary behavior timeseries.")
            return load_data.load_binary_timeseries_df(binary_timeseries_file_path)


    def _load_or_compute_crosscorr_df(self):
        crosscorr_file_path = os.path.join(self.params['processed_data_dir'], 'inter_agent_crosscorrelation_df.pkl')
        if 1:
            self.logger.info("Computing inter-agent cross-correlations for all behavioral types.")
            crosscorrelation_df_between_all_m1_amd_m2_behavior = analyze_data.compute_behavioral_cross_correlations(
                self.binary_behav_timeseries_df, self.params, shuffled=False, serial=False, remake_results=True)
            crosscorrelation_df_between_all_m1_amd_m2_behavior.to_pickle(crosscorr_file_path)
            self.logger.info(f"Saved interagent cross-correlation df to: {crosscorr_file_path}")
            return crosscorrelation_df_between_all_m1_amd_m2_behavior
        else:
            return load_data.load_binary_crosscorr_df(crosscorr_file_path)



    def _load_or_compute_binary_timeseries_auto_and_crosscorr(self):
        """Load or compute binary timeseries autocorrelation DataFrame."""
        autocorr_file_path = os.path.join(self.params['processed_data_dir'], 'scaled_autocorrelations.pkl')
        if self.params.get('remake_scaled_autocorr', False) or not os.path.exists(autocorr_file_path):
            self.logger.info("Computing scaled autocorrelations.")
            use_parallel = self.params['use_parallel']
            # use_parallel = False
            auto_and_cross_corr_df = analyze_data.calculate_auto_and_cross_corrs_bet_behav_vectors(
                self.binary_behav_timeseries_df,
                self.num_cpus,
                use_parallel=use_parallel)
            auto_and_cross_corr_df.to_pickle(autocorr_file_path)
            return auto_and_cross_corr_df
        else:
            self.logger.info("Loading existing scaled autocorrelations.")
            return load_data.load_binary_autocorr_df(autocorr_file_path)


    def _load_or_compute_neural_fr_timeseries_df(self):
        """Load or compute neural firing rate timeseries DataFrame."""
        neural_timeseries_df_file_path = os.path.join(self.params['processed_data_dir'], 'neural_timeseries_df.pkl')
        if self.params.get('remake_neural_timeseries', False) or not os.path.exists(neural_timeseries_df_file_path):
            self.logger.info("Generating neural firing rate timeseries.")
            neural_fr_df = curate_data.make_binned_unit_fr_df_for_each_run(
                self.spike_times_df, self.nan_removed_gaze_data_df, self.params
            )
            neural_fr_df.to_pickle(neural_timeseries_df_file_path)
            return neural_fr_df
        else:
            self.logger.info("Loading existing neural firing rate timeseries.")
            return load_data.load_neural_timeseries_df(neural_timeseries_df_file_path)


    def plot_data(self):
        out_dir = self.params.get('processed_data_dir')
        plotter.process_and_plot_behaviors(self.binary_behav_timeseries_df, out_dir)
        # plotter.plot_random_run_snippets(self.neural_fr_timeseries_df)
        # plotter.plot_fixations_and_saccades(self.synchronized_gaze_data_df, self.fixation_df, self.saccade_df, self.params)
        # plotter.plot_mean_auto_and_crosscorrelations_for_monkey_pairs(self.recording_sessions_and_monkeys, self.binary_timeseries_scaled_auto_and_crosscorr_df, self.params)
        # plotter.plot_auto_and_cross_correlations(self.binary_timeseries_scaled_auto_and_crosscorr_df, self.params)




    def calculate_firing_rate_statistics(self):
        """
        Calculate the average and standard deviation of firing rates in Hz for different regions.

        Returns:
            firing_rate_stats_df (pd.DataFrame): A dataframe containing regions, 
                                                average firing rates, and standard deviations.
        """
        # Extract relevant columns
        spike_times_df = self.spike_times_df.copy()

        # Calculate the firing rate for each unit
        def compute_firing_rate(spike_times):
            if len(spike_times) < 2:
                return 0  # Avoid division by zero for units with less than two spikes
            duration = spike_times[-1] - spike_times[0]  # Time range in seconds
            return len(spike_times) / duration if duration > 0 else 0

        # Ensure spike_ts is a list of floats for calculations
        spike_times_df['spike_rate'] = spike_times_df['spike_ts'].apply(
            lambda spike_ts: compute_firing_rate(sorted(spike_ts))
        )

        # Group by region and calculate mean and std for firing rates
        region_stats = (
            spike_times_df.groupby('region')['spike_rate']
            .agg(['mean', 'std'])
            .reset_index()
            .rename(columns={'mean': 'avg_firing_rate', 'std': 'std_firing_rate'})
        )

        return region_stats


    def run(self):
        """Runs the data processing steps in sequence."""
        self.populate_params_with_data_paths()
        # self.create_and_submit_shuffled_cross_corr_jobs()
        # self.get_data()
        # firing_rate_stats_df = self.calculate_firing_rate_statistics()
        # print(firing_rate_stats_df)

        self.prune_data()
        self.analyze_behavior()
        # self.plot_data()cat