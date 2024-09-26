#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14::51:42 2024

@author: pg496
"""

import logging
import os
import multiprocessing

import util
import curate_data
import load_data
import fix_and_saccades

import pdb


def check_non_interactive_data(gaze_data_dict):
    """
    Checks if any 'non_interactive' key in the gaze data dictionary has data and identifies which runs contain data.
    Parameters:
    - gaze_data_dict (dict): A dictionary structured with sessions as top-level keys, followed by 
      'interactive' and 'non_interactive' keys, which contain run numbers and their associated data.
    Returns:
    - results (dict): A dictionary summarizing the sessions and runs where 'non_interactive' data is found,
      including details on the data keys ('positions', 'pupil_size', 'neural_timeline') and subkeys ('m1', 'm2').
    """
    results = {}
    # Iterate over each session in the gaze data dictionary
    for session_name, interaction_types in gaze_data_dict.items():
        # Check if 'non_interactive' exists in the interaction types
        if 'non_interactive' in interaction_types:
            # Iterate over each run in the 'non_interactive' interaction type
            for run_number, run_data in interaction_types['non_interactive'].items():
                # Initialize a flag to track if this run has any data
                has_data = False
                run_results = {}
                # Check positions data for m1 and m2
                if 'positions' in run_data:
                    positions = run_data['positions']
                    m1_data = positions.get('m1')
                    m2_data = positions.get('m2')
                    m1_has_data = m1_data is not None and m1_data.size > 0
                    m2_has_data = m2_data is not None and m2_data.size > 0
                    if m1_has_data or m2_has_data:
                        run_results['positions'] = {
                            'm1': m1_has_data,
                            'm2': m2_has_data
                        }
                        has_data = True
                # Check pupil size data for m1 and m2
                if 'pupil_size' in run_data:
                    pupil_size = run_data['pupil_size']
                    m1_data = pupil_size.get('m1')
                    m2_data = pupil_size.get('m2')
                    m1_has_data = m1_data is not None and m1_data.size > 0
                    m2_has_data = m2_data is not None and m2_data.size > 0
                    if m1_has_data or m2_has_data:
                        run_results['pupil_size'] = {
                            'm1': m1_has_data,
                            'm2': m2_has_data
                        }
                        has_data = True
                # Check neural timeline data
                if 'neural_timeline' in run_data and run_data['neural_timeline'] is not None and run_data['neural_timeline'].size > 0:
                    run_results['neural_timeline'] = True
                    has_data = True
                # If any data was found, add it to the results
                if has_data:
                    if session_name not in results:
                        results[session_name] = {}
                    results[session_name][run_number] = run_results
    # Log and return the results
    if results:
        print("Found non_interactive data in the following sessions and runs:")
        for session, runs in results.items():
            for run, data_keys in runs.items():
                print(f"Session: {session}, Run: {run}, Data: {data_keys}")
    else:
        print("No non_interactive data found in the gaze data dictionary.")
    return results



class DataManager:
    def __init__(self, params):
        self.setup_logger()
        self.params = params
        self.find_n_cores()
        self.initialize_class_objects()


    def setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)


    def find_n_cores(self):
        """Determine the number of CPU cores available, prioritizing SLURM if available."""
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
        """Initialize class object attributes."""
        self.gaze_data_dict = None
        self.empty_gaze_dict_paths = None
        self.nan_removed_gaze_data_dict = None

        self.fixation_dict = None
        self.saccade_dict = None


    def populate_params_with_data_paths(self):
        self.params = curate_data.add_root_data_to_params(self.params)
        self.params = curate_data.add_processed_data_to_params(self.params)
        self.params = curate_data.add_raw_data_dir_to_params(self.params)
        self.params = curate_data.add_paths_to_all_data_files_to_params(self.params)
        self.params = curate_data.prune_data_file_paths(self.params)


    def get_data(self):
        """
        Loads gaze data into a dictionary format from the available position, time, and pupil size files.
        """
        # Define paths to save/load the variables
        processed_data_dir = self.params['processed_data_dir']
        gaze_data_file_path = os.path.join(processed_data_dir, 'gaze_data_dict.pkl')
        missing_data_file_path = os.path.join(processed_data_dir, 'missing_data_dict_paths.pkl')
        # Use the compute_or_load_variables function to compute or load the gaze data
        self.gaze_data_dict, self.empty_gaze_dict_paths = util.compute_or_load_variables(
            compute_func=curate_data.make_gaze_data_dict,
            load_func=load_data.get_gaze_data_dict,  # Function to load the data, to be implemented next
            file_paths=[gaze_data_file_path, missing_data_file_path],
            remake_flag_key='remake_gaze_data_dict',
            params=self.params  # Pass the params dictionary
        )


    def prune_data(self):
        self.nan_removed_gaze_data_dict = curate_data.prune_nan_values_in_timeseries(self.gaze_data_dict, self.params)


    def analyze_behavior(self):
        # Detect fixations and saccades for m1
        self.fixation_dict, self.saccade_dict = fix_and_saccades.detect_fixations_and_saccades(
            self.nan_removed_gaze_data_dict, params=self.params
        )


    def run(self):
        self.populate_params_with_data_paths()
        self.get_data()
        self.prune_data()
        self.analyze_behavior()





# if params.get('extract_postime_from_mat_files', False):
#     sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
#         filter_behav.get_gaze_timepos_across_sessions(params)
# else:
#     sorted_position_path_list, m1_gaze_positions, m2_gaze_positions, sorted_time_path_list, time_vecs = \
#         load_data.get_combined_gaze_pos_and_time_lists(params)

# print(sorted_position_path_list)


# pos_pattern = r"(\d{8})_position_(\d+).mat"
# session_infos = [{'session_name': re.match(pos_pattern, os.path.basename(f)).group(1), 'run_number': int(re.match(pos_pattern, os.path.basename(f)).group(2))} for f in sorted_position_path_list]

# # Extract fixations and saccades for both monkeys
# fixations_m1, saccades_m1, fixations_m2, saccades_m2 = fix_and_saccades.extract_fixations_for_both_monkeys(params, m1_gaze_positions, m2_gaze_positions, time_vecs, session_infos)