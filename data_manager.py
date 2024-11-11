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
        self.nan_removed_gaze_data_df = None
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
        self.gaze_data_df, self.missing_data_paths = self._load_or_compute_gaze_data()
        self.gaze_data_df = self._filter_sessions_with_ephys(self.gaze_data_df)
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
        nan_removed_gaze_data_file_path = os.path.join(self.params['processed_data_dir'], 'nan_removed_gaze_data_df.pkl')
        if self.params.get('remake_nan_removed_gaze_data_df', False) or not os.path.exists(nan_removed_gaze_data_file_path):
            self.logger.info("Pruning NaN values in gaze data.")
            nan_removed_gaze_data = curate_data.prune_nan_values_in_timeseries(self.gaze_data_df, self.params)
            nan_removed_gaze_data.to_pickle(nan_removed_gaze_data_file_path)
            self.nan_removed_gaze_data_df = nan_removed_gaze_data
        else:
            self.logger.info("Loading pruned gaze data.")
            self.nan_removed_gaze_data_df = load_data.get_nan_removed_gaze_data_df(nan_removed_gaze_data_file_path)


    def analyze_behavior(self):
        """Analyze behavior by detecting fixations and saccades and computing binary timeseries and autocorrelations."""
        self.fixation_df, self.saccade_df = self._load_or_compute_fixations_and_saccades()
        self.add_saccade_rois_in_dataframe()
        self.binary_behav_timeseries_df = self._load_or_compute_binary_behav_timeseries()
        self.binary_timeseries_scaled_autocorr_df = self._load_or_compute_binary_timeseries_autocorr()
        self.neural_fr_timeseries_df = self._load_or_compute_neural_fr_timeseries_df()
        pdb.set_trace()


    def _load_or_compute_fixations_and_saccades(self):
        """Load or compute fixation and saccade DataFrames."""
        fixation_file_path = os.path.join(self.params['processed_data_dir'], 'fixation_df.pkl')
        saccade_file_path = os.path.join(self.params['processed_data_dir'], 'saccade_df.pkl')
        if self.params.get('remake_fix_and_sacc', False) or not (os.path.exists(fixation_file_path) and os.path.exists(saccade_file_path)):
            self.logger.info("Detecting fixations and saccades.")
            fixation_df, saccade_df = fix_and_saccades.detect_fixations_and_saccades(self.nan_removed_gaze_data_df, self.params)
            fixation_df.to_pickle(fixation_file_path)
            saccade_df.to_pickle(saccade_file_path)
            return fixation_df, saccade_df
        else:
            self.logger.info("Loading existing fixation and saccade data.")
            return load_data.load_fixation_and_saccade_dfs(fixation_file_path, saccade_file_path)

    def add_saccade_rois_in_dataframe(self):
        # Initialize lists to store 'from' and 'to' labels for each row
        all_from_labels = []
        all_to_labels = []
        # Iterate over each row in the saccade dataframe
        for _, saccade_row in self.saccade_df.iterrows():
            session_name = saccade_row['session_name']
            interaction_type = saccade_row['interaction_type']
            run_number = saccade_row['run_number']
            agent = saccade_row['agent']
            saccades = saccade_row['saccade_start_stop']
            # Filter the gaze data to match the current row's session, interaction type, run, and agent
            gaze_row = self.nan_removed_gaze_data_df[
                (self.nan_removed_gaze_data_df['session_name'] == session_name) &
                (self.nan_removed_gaze_data_df['interaction_type'] == interaction_type) &
                (self.nan_removed_gaze_data_df['run_number'] == run_number) &
                (self.nan_removed_gaze_data_df['agent'] == agent)
            ].iloc[0]  # Should match only one row
            positions = gaze_row['positions']
            roi_rects = gaze_row['roi_rects']
            # Temporary lists for 'from' and 'to' labels for the current row
            from_labels = []
            to_labels = []
            # Process each saccade to identify 'from' and 'to' ROIs
            for start_stop in saccades:
                start_idx, stop_idx = start_stop
                # Get starting and ending positions of the saccade
                start_pos = positions[start_idx]
                stop_pos = positions[stop_idx]
                # Determine 'from' ROI
                from_roi = 'elsewhere'
                for roi_name, rect in roi_rects.items():
                    if rect[0] <= start_pos[0] <= rect[2] and rect[1] <= start_pos[1] <= rect[3]:
                        from_roi = roi_name
                        break
                # Determine 'to' ROI
                to_roi = 'elsewhere'
                for roi_name, rect in roi_rects.items():
                    if rect[0] <= stop_pos[0] <= rect[2] and rect[1] <= stop_pos[1] <= rect[3]:
                        to_roi = roi_name
                        break
                # Append 'from' and 'to' labels for the current saccade
                from_labels.append(from_roi)
                to_labels.append(to_roi)
            # Append the lists of labels for the current row to the main lists
            all_from_labels.append(from_labels)
            all_to_labels.append(to_labels)
        # Add 'from' and 'to' columns to the saccade dataframe
        self.saccade_df['from'] = all_from_labels
        self.saccade_df['to'] = all_to_labels

    def add_fixation_location_in_dataframe(self):
        # Initialize a list to store 'location' labels for each row
        all_location_labels = []
        # Iterate over each row in the fixation dataframe
        for _, fixation_row in self.fixation_df.iterrows():
            session_name = fixation_row['session_name']
            interaction_type = fixation_row['interaction_type']
            run_number = fixation_row['run_number']
            agent = fixation_row['agent']
            fixations = fixation_row['fixation_start_stop']
            # Filter the gaze data to match the current row's session, interaction type, run, and agent
            gaze_row = self.nan_removed_gaze_data_df[
                (self.nan_removed_gaze_data_df['session_name'] == session_name) &
                (self.nan_removed_gaze_data_df['interaction_type'] == interaction_type) &
                (self.nan_removed_gaze_data_df['run_number'] == run_number) &
                (self.nan_removed_gaze_data_df['agent'] == agent)
            ].iloc[0]  # Should match only one row
            positions = gaze_row['positions']
            roi_rects = gaze_row['roi_rects']
            # Temporary list for 'location' labels for the current row
            location_labels = []
            # Process each fixation to identify its 'location' ROI
            for start_stop in fixations:
                start_idx, stop_idx = start_stop
                # Calculate the mean position of the fixation
                fixation_positions = positions[start_idx:stop_idx+1]
                mean_position = [
                    sum(pos[0] for pos in fixation_positions) / len(fixation_positions),
                    sum(pos[1] for pos in fixation_positions) / len(fixation_positions)
                ]
                # Determine 'location' ROI
                location = 'elsewhere'
                for roi_name, rect in roi_rects.items():
                    if rect[0] <= mean_position[0] <= rect[2] and rect[1] <= mean_position[1] <= rect[3]:
                        location = roi_name
                        break
                # Append the 'location' label for the current fixation
                location_labels.append(location)
            # Append the list of location labels for the current row
            all_location_labels.append(location_labels)
        # Add 'location' column to the fixation dataframe
        self.fixation_df['location'] = all_location_labels


    def _load_or_compute_binary_behav_timeseries(self):
        """Load or compute binary behavior timeseries DataFrame."""
        binary_timeseries_file_path = os.path.join(self.params['processed_data_dir'], 'binary_behav_timeseries.pkl')
        if self.params.get('remake_binary_timeseries', False) or not os.path.exists(binary_timeseries_file_path):
            self.logger.info("Generating binary behavior timeseries.")
            binary_timeseries_df = analyze_data.create_binary_behav_timeseries_df(
                self.fixation_df, self.saccade_df, self.nan_removed_gaze_data_df, self.params
            )
            binary_timeseries_df.to_pickle(binary_timeseries_file_path)
            return binary_timeseries_df
        else:
            self.logger.info("Loading existing binary behavior timeseries.")
            return load_data.load_binary_timeseries_df(binary_timeseries_file_path)


    def _load_or_compute_binary_timeseries_autocorr(self):
        """Load or compute binary timeseries autocorrelation DataFrame."""
        autocorr_file_path = os.path.join(self.params['processed_data_dir'], 'scaled_autocorrelations.pkl')
        if self.params.get('remake_scaled_autocorr', False) or not os.path.exists(autocorr_file_path):
            self.logger.info("Computing scaled autocorrelations.")
            autocorr_df = analyze_data.compute_scaled_autocorrelations_for_behavior_df(self.binary_behav_timeseries_df, self.params)
            autocorr_df.to_pickle(autocorr_file_path)
            return autocorr_df
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
        plotter.plot_random_run_snippets(self.neural_fr_timeseries_df)


    def run(self):
        """Runs the data processing steps in sequence."""
        self.populate_params_with_data_paths()
        self.get_data()
        self.prune_data()
        self.analyze_behavior()
        self.plot_data()