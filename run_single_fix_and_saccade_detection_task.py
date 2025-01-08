import pickle
import sys
import os
import logging
import pandas as pd

import load_data
import detect_eye_mvm_behav_from_gaze_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task_execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main(task_key, params_file_path):
    """
    Main function to run fixation and saccade detection for a single task.
    Parameters:
    - task_key (str): Key arguments to fetch the specific run from the gaze data dictionary.
    - params_file_path (str): Path to the serialized params file.
    """
    logger.info("Starting the main function")
    # Load params to get the processed data directory path
    try:
        with open(params_file_path, 'rb') as f:
            params = pickle.load(f)
        logger.info("Parameters loaded successfully from %s", params_file_path)
    except Exception as e:
        logger.error("Failed to load parameters from %s: %s", params_file_path, e)
        sys.exit(1)
    # Paths to gaze data dictionary and missing data paths
    processed_data_dir = params['processed_data_dir']
    sparse_nan_removed_sync_gaze_data_df_path = os.path.join(processed_data_dir, 'sparse_nan_removed_sync_gaze_data_df.pkl')
    # Load gaze data dictionary
    try:
        sparse_nan_removed_sync_gaze_data_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_path)
        logger.info("Gaze data dataframe loaded successfully from %s", sparse_nan_removed_sync_gaze_data_df_path)
    except Exception as e:
        logger.error("Failed to load gaze data from %s: %s", sparse_nan_removed_sync_gaze_data_df_path, e)
        sys.exit(1)
    # Parse task_key to fetch the required positions array
    try:
        session, interaction_type, run, agent = task_key.split(',')
        session = session.strip()
        interaction_type = interaction_type.strip()
        run = int(run.strip())  # Ensure run is an integer
        agent = agent.strip()
        logger.info(
            "Parsed task key successfully: session=%s, interaction_type=%s, run=%s, agent=%s",
            session, interaction_type, run, agent
        )
    except Exception as e:
        logger.error("Failed to parse task key %s: %s", task_key, e)
        sys.exit(1)
    # Filter the DataFrame to get the matching row
    try:
        filtered_df = sparse_nan_removed_sync_gaze_data_df[
            (sparse_nan_removed_sync_gaze_data_df['session_name'] == session) &
            (sparse_nan_removed_sync_gaze_data_df['interaction_type'] == interaction_type) &
            (sparse_nan_removed_sync_gaze_data_df['run_number'] == run) &
            (sparse_nan_removed_sync_gaze_data_df['agent'] == agent)
        ]
        if filtered_df.empty:
            logger.error("No data found for task key: %s", task_key)
            sys.exit(1)
        positions = filtered_df['positions'].iloc[0]  # Assuming 'positions' is a single column per row
        logger.info("Successfully extracted positions for fixation and saccade detection.")
        # Run fixation and saccade detection
        fix_indices, sacc_indices, microsacc_indices = \
            detect_eye_mvm_behav_from_gaze_data._detect_fixations_saccades_and_microsaccades_in_run(positions, session)
        logger.info("Fixation and saccade detection completed successfully.")
    except Exception as e:
        logger.error("Failed during fixation and saccade detection: %s", e)
        sys.exit(1)
    # Save results
    try:
        hpc_data_subfolder = params.get('hpc_job_output_subfolder', '')
        fixation_output_path = os.path.join(processed_data_dir, hpc_data_subfolder, f'fixation_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')
        saccade_output_path = os.path.join(processed_data_dir, hpc_data_subfolder, f'saccade_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')
        os.makedirs(os.path.dirname(fixation_output_path), exist_ok=True)
        with open(fixation_output_path, 'wb') as f:
            pickle.dump(fix_indices, f)
        logger.info("Fixation results saved successfully at %s", fixation_output_path)
        with open(saccade_output_path, 'wb') as f:
            pickle.dump((sacc_indices, microsacc_indices), f)
        logger.info("Saccade results saved successfully at %s", saccade_output_path)
    except Exception as e:
        logger.error("Failed to save results: %s", e)
        sys.exit(1)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Incorrect number of arguments. Usage: python script.py <task_key> <params_file_path>")
        sys.exit(1)
    task_key = sys.argv[1]
    params_file_path = sys.argv[2]
    main(task_key, params_file_path)
