
import numpy as np
import pandas as pd
import os
import logging
from itertools import chain

import load_data
import curate_data

import pdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the main function")
    # Initialize params with data paths
    params = _initialize_params()
    ## Get the gaze data
    sparse_nan_removed_sync_gaze_data_df_filepath = os.path.join(params['processed_data_dir'], 'sparse_nan_removed_sync_gaze_data_df.pkl')
    logger.info("Loading sparse_nan_removed_sync_gaze_df")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)

    eye_mvm_behav_df_file_path = os.path.join(params['processed_data_dir'], 'eye_mvm_behav_df.pkl')
    logger.info("Loading eye_mvm_behav_df")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)

    binary_behavior_timeseries_df = _create_binary_behavior_timeseries_df(sparse_nan_removed_sync_gaze_df, eye_mvm_behav_df)

    pdb.set_trace()

    return 0


def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'xx': False,
        'is_grace': False
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    params = curate_data.add_raw_data_dir_to_params(params)
    params = curate_data.add_paths_to_all_data_files_to_params(params)
    params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(params)
    logger.info("Parameters initialized successfully")
    return params



def _create_binary_behavior_timeseries_df(position_df, behavior_df):
    """
    Generate a binary timeseries dataframe for fixations and saccades.
    Parameters:
    - position_df (pd.DataFrame): Dataframe containing position data.
    - behavior_df (pd.DataFrame): Dataframe containing behavioral data.
    Returns:
    - binary_df (pd.DataFrame): Dataframe containing binary timeseries for behaviors.
    """
    binary_rows = []
    logger.info(f"Starting processing of {len(position_df)} position rows.")
    for session_name, session_group in position_df.groupby(['session_name', 'interaction_type', 'run_number', 'agent']):
        logger.info(f"Processing session group: {session_name}")
        pos_group = session_group.iloc[0]
        behav_row = behavior_df[(behavior_df['session_name'] == pos_group['session_name']) &
                                (behavior_df['interaction_type'] == pos_group['interaction_type']) &
                                (behavior_df['run_number'] == pos_group['run_number']) &
                                (behavior_df['agent'] == pos_group['agent'])]
        if behav_row.empty:
            logger.warning(f"No behavioral data found for session: {session_name}")
            continue
        behav_row = behav_row.iloc[0]
        position_length = len(pos_group['positions'])
        # Generate binary vectors for fixations
        fixation_start_stop = behav_row['fixation_start_stop']
        fixation_locations = __merge_objects_in_location_array(behav_row['fixation_location'])
        unique_fixation_locations = set(chain.from_iterable(fixation_locations))
        for location in unique_fixation_locations:
            binary_vector = np.zeros(position_length, dtype=int)
            for (start, stop), loc in zip(fixation_start_stop, fixation_locations):
                if location in loc:
                    binary_vector[start:stop + 1] = 1
            binary_rows.append({
                'session_name': pos_group['session_name'],
                'interaction_type': pos_group['interaction_type'],
                'run_number': pos_group['run_number'],
                'agent': pos_group['agent'],
                'behavior_type': 'fixation',
                'location': location,
                'binary_vector': binary_vector
            })
        # Generate binary vectors for saccades (based on `from` and `to` locations)
        saccade_start_stop = behav_row['saccade_start_stop']
        saccade_from = __merge_objects_in_location_array(behav_row['saccade_from'])
        saccade_to = __merge_objects_in_location_array(behav_row['saccade_to'])
        unique_from_locations = set(chain.from_iterable(saccade_from))
        unique_to_locations = set(chain.from_iterable(saccade_to))
        for location in unique_from_locations:
            binary_vector = np.zeros(position_length, dtype=int)
            for (start, stop), from_loc in zip(saccade_start_stop, saccade_from):
                if location in from_loc:
                    binary_vector[start:stop + 1] = 1
            binary_rows.append({
                'session_name': pos_group['session_name'],
                'interaction_type': pos_group['interaction_type'],
                'run_number': pos_group['run_number'],
                'agent': pos_group['agent'],
                'behavior_type': 'saccade_from',
                'location': location,
                'binary_vector': binary_vector
            })
        for location in unique_to_locations:
            binary_vector = np.zeros(position_length, dtype=int)
            for (start, stop), to_loc in zip(saccade_start_stop, saccade_to):
                if location in to_loc:
                    binary_vector[start:stop + 1] = 1
            binary_rows.append({
                'session_name': pos_group['session_name'],
                'interaction_type': pos_group['interaction_type'],
                'run_number': pos_group['run_number'],
                'agent': pos_group['agent'],
                'behavior_type': 'saccade_to',
                'location': location,
                'binary_vector': binary_vector
            })
    logger.info("Finished processing all sessions.")
    # Create a DataFrame from the binary rows
    binary_df = pd.DataFrame(binary_rows)
    logger.info(f"Generated binary behavior dataframe with {len(binary_df)} rows.")
    return binary_df

# Define helper function to handle "object" location merging
def __merge_objects_in_location_array(location_array):
    """
    Merges all 'left_nonsocial_object' and 'right_nonsocial_object' into 'object'.
    
    Parameters:
    - location_array (list): A list of location strings or lists of location strings.
    
    Returns:
    - merged_location_array (list): A list with nonsocial objects merged into 'object'.
    """
    merged_location_array = []
    for loc in location_array:
        if isinstance(loc, list):  # If loc is a list, process each element
            merged_location_array.append(['object' if 'object' in subloc else subloc for subloc in loc])
        else:  # If loc is a single string
            merged_location_array.append('object' if 'object' in loc else loc)
    return merged_location_array



if __name__ == "__main__":
    main()