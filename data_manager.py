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

import util
import curate_data
import load_data
import fix_and_saccades
import analyze_data
import plotter

import pdb



class DataManager:
    def __init__(self, params):
        """Initialize the DataManager with parameters, logger setup, and CPU detection."""
        self.setup_logger()
        self.params = params
        self.find_n_cores()
        self.initialize_class_objects()


    def setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)


    def find_n_cores(self):
        """Determine the number of CPU cores available, prioritizing SLURM if available, 
        and update params with the detected core count."""
        try:
            slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
            num_cpus = int(slurm_cpus)
            self.logger.info(f"SLURM detected {num_cpus} CPUs")
        except Exception as e:
            self.logger.warning(f"Failed to detect cores with SLURM_CPUS_ON_NODE: {e}")
            num_cpus = None
        if num_cpus is None or num_cpus <= 1:
            num_cpus = multiprocessing.cpu_count()
            self.logger.info(f"Multiprocessing detected {num_cpus} CPUs")
        os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus)
        self.num_cpus = num_cpus
        self.params['num_cpus'] = num_cpus
        self.logger.info(f"NumExpr set to use {num_cpus} threads")


    def initialize_class_objects(self):
        """Initialize class object attributes to None."""
        self.recording_sessions_and_monkeys = None
        self.gaze_data_df = None
        self.spike_times_df = None
        self.missing_data_paths = None
        self.nan_removed_gaze_data_df = None
        self.fixation_df = None
        self.saccade_df = None
        self.binary_behav_timeseries_df = None
        self.binary_timeseries_scaled_autocorr_df = None


    def populate_params_with_data_paths(self):
        """Populate the params dictionary with various data paths needed for processing."""
        self.params = curate_data.add_root_data_to_params(self.params)
        self.params = curate_data.add_processed_data_to_params(self.params)
        self.params = curate_data.add_raw_data_dir_to_params(self.params)
        self.params = curate_data.add_paths_to_all_data_files_to_params(self.params)
        self.params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(self.params)


    def get_data(self):
        """
        Load or compute necessary gaze and spike data, filtering for sessions with electrophysiology data.
        """
        # Load session list for ephys data
        self.recording_sessions_and_monkeys = self._load_ephys_sessions()
        # Load or compute gaze data and missing data paths
        self.gaze_data_df, self.missing_data_paths = self._load_or_compute_gaze_data()
        self.gaze_data_df = util.extract_df_rows_for_sessions_with_ephys(self.gaze_data_df, self.recording_sessions_and_monkeys)
        # Load or compute spike times DataFrame
        self.spike_times_df = self._load_or_compute_spike_times()
        self.spike_times_df = util.extract_df_rows_for_sessions_with_ephys(self.spike_times_df, self.recording_sessions_and_monkeys)


    def _load_ephys_sessions(self):
        """Load the list of sessions with ephys data."""
        path_to_ephys_sessions = os.path.join(self.params['processed_data_dir'], 'ephys_days_and_monkeys.pkl')
        return load_data.load_recording_days(path_to_ephys_sessions)


    def _load_or_compute_gaze_data(self):
        """Helper to load or compute gaze data and missing paths DataFrames."""
        gaze_data_file_path = os.path.join(self.params['processed_data_dir'], 'gaze_data_df.pkl')
        missing_data_paths_file_path = os.path.join(self.params['processed_data_dir'], 'missing_data_paths.pkl')
        return util.compute_or_load_variables(
            curate_data.make_gaze_data_df,                           # Compute function
            load_data.get_gaze_data_df,                              # Load function
            [gaze_data_file_path, missing_data_paths_file_path],     # File paths
            'remake_gaze_data_df',                                   # Remake flag key
            self.params,                                             # Params for load
            self.params                                              # Params for compute
        )


    def _load_or_compute_spike_times(self):
        """Helper to load or compute spike times DataFrame."""
        spike_times_file_path = os.path.join(self.params['processed_data_dir'], 'spike_times_df.pkl')
        return util.compute_or_load_variables(
            curate_data.make_spike_times_df,                         # Compute function
            load_data.get_spike_times_df,                            # Load function
            spike_times_file_path,                                   # File path
            'remake_spike_times_df',                                 # Remake flag key
            self.params,                                             # Params for load
            self.params                                              # Params for compute
        )


    def prune_data(self):
        """Prune NaN values from gaze data using compute_or_load_variables."""
        nan_removed_gaze_data_file_path = os.path.join(self.params['processed_data_dir'], 'nan_removed_gaze_data_df.pkl')
        # Load or compute pruned gaze data
        self.nan_removed_gaze_data_df = util.compute_or_load_variables(
            curate_data.prune_nan_values_in_timeseries,     # Compute function
            load_data.get_nan_removed_gaze_data_df,         # Load function
            nan_removed_gaze_data_file_path,                # File path
            'remake_nan_removed_gaze_data_df',            # Remake flag key
            self.params,                                    # Params
            self.gaze_data_df                               # Gaze data as argument for compute function
        )


    def analyze_behavior(self):
        """Analyze behavior by detecting fixations and saccades, and compute binary timeseries and autocorrelations."""
        # Load or compute fixation and saccade DataFrames
        self.fixation_df, self.saccade_df = self._load_or_compute_fixations_and_saccades()
        # Load or compute binary behavior timeseries DataFrame
        self.binary_behav_timeseries_df = self._load_or_compute_binary_behav_timeseries()
        # Load or compute binary timeseries autocorrelation DataFrame
        self.binary_timeseries_scaled_autocorr_df = self._load_or_compute_binary_timeseries_autocorr()
        self.binned_neural_timeseries_df = self._load_or_compute_neural_fr_timeseries_df()


    def _load_or_compute_fixations_and_saccades(self):
        """Helper to load or compute fixation and saccade DataFrames."""
        fixation_file_path = os.path.join(self.params['processed_data_dir'], 'fixation_df.pkl')
        saccade_file_path = os.path.join(self.params['processed_data_dir'], 'saccade_df.pkl')
        return util.compute_or_load_variables(
            fix_and_saccades.detect_fixations_and_saccades,     # Compute function
            load_data.load_fixation_and_saccade_dfs,            # Load function
            [fixation_file_path, saccade_file_path],            # File paths
            'remake_fix_and_sacc',                              # Remake flag key
            self.params,                                        # Params
            self.nan_removed_gaze_data_df,                      # Gaze data as argument
            self.params                                         # Params again as argument
        )


    def _load_or_compute_binary_behav_timeseries(self):
        """Helper to load or compute binary behavior timeseries DataFrame."""
        binary_timeseries_file_path = os.path.join(self.params['processed_data_dir'], 'binary_behav_timeseries.pkl')
        return util.compute_or_load_variables(
            analyze_data.create_binary_behav_timeseries_df,      # Compute function
            load_data.load_binary_timeseries_df,                 # Load function
            binary_timeseries_file_path,                         # File path
            'remake_binary_timeseries',                          # Remake flag key
            self.params,                                         # Params
            self.fixation_df,                                    # Fixation data
            self.saccade_df,                                     # Saccade data
            self.nan_removed_gaze_data_df,                       # Gaze data
            self.params                                          # Params
        )


    def _load_or_compute_binary_timeseries_autocorr(self):
        """Helper to load or compute binary timeseries autocorrelation DataFrame."""
        autocorr_file_path = os.path.join(self.params['processed_data_dir'], 'scaled_autocorrelations.pkl')
        return util.compute_or_load_variables(
            analyze_data.compute_scaled_autocorrelations_for_behavior_df,  # Compute function
            load_data.load_binary_autocorr_df,                             # Load function
            autocorr_file_path,                                            # File path
            'remake_scaled_autocorr',                                      # Remake flag key
            self.params,                                                   # Params
            self.binary_behav_timeseries_df,
            self.params                                             
        )


    def _load_or_compute_neural_fr_timeseries_df(self):
        """Helper to load or compute binary timeseries autocorrelation DataFrame."""
        neural_timeseries_df_file_path = os.path.join(self.params['processed_data_dir'], 'neural_timeseries_df.pkl')
        return util.compute_or_load_variables(
            curate_data.make_binned_unit_fr_df_for_each_run,
            load_data.load_neural_timeseries_df,
            neural_timeseries_df_file_path,
            'remake_neural_timeseries',
            self.params,
            self.spike_times_df,
            self.nan_removed_gaze_data_df_ephys_days,
            self.params
        )


    def plot_behavior(self):
        plotter.plot_fixations_and_saccades(self.nan_removed_gaze_data_df, self.fixation_df, self.saccade_df, self.params)
        # plot scaled autocorrelations
        """
        for each session, interaction type and run, plot out the scaled autocorrelation vs lag
        for 5 seconds lag for all columns of behav autocorr dataframe
        """


    def run(self):
        """Runs the data processing steps in sequence."""
        #!! ROI rects need to be offset adjusted. Do positions need remapping as well?
        self.populate_params_with_data_paths()
        self.get_data()
        self.prune_data()
        util.verify_presence_of_recording_sessions(self.recording_sessions_and_monkeys, self.spike_times_df)
        self.nan_removed_gaze_data_df_ephys_days = util.extract_df_rows_for_sessions_with_ephys(self.nan_removed_gaze_data_df, self.recording_sessions_and_monkeys)
        self.analyze_behavior()
        pdb.set_trace()
        return 0