import pickle
import sys
import os
import logging
import load_data
import fix_and_saccades

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


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
        logger.info("Parameters loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load parameters from {params_file_path}: {e}")
        sys.exit(1)
    # Paths to gaze data dictionary and missing data paths
    processed_data_dir = params['processed_data_dir']
    nan_removed_gaze_data_dict_path = os.path.join(processed_data_dir, 'nan_removed_gaze_data_dict.pkl')
    # Load gaze data dictionary
    try:
        nan_removed_gaze_data_dict_path = load_data.get_nan_removed_gaze_data_dict(nan_removed_gaze_data_dict_path)
        logger.info("Gaze data dictionary loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load gaze data from {nan_removed_gaze_data_dict_path}: {e}")
        sys.exit(1)
    # Parse task_key to fetch the required positions array
    try:
        session, interaction_type, run, agent = task_key.split(',')
        session = session.strip()
        interaction_type = interaction_type.strip()
        run = int(run.strip())
        agent = agent.strip()
        logger.info(f"Parsed task key successfully: session: {session}; interaction_type: {interaction_type}; run: {str(run)}; agent: {agent}")
    except Exception as e:
        logger.error(f"Failed to parse task key {task_key}: {e}")
        sys.exit(1)
    # Prepare task arguments
    task_args = (session, interaction_type, run, agent, params)
    # Run fixation and saccade detection
    try:
        fix_dict, sacc_dict = fix_and_saccades.process_fix_and_saccade_for_specific_run(task_args, nan_removed_gaze_data_dict_path)
        logger.info("Fixation and saccade detection completed successfully.")
    except Exception as e:
        logger.error(f"Failed during fixation and saccade detection: {e}")
        sys.exit(1)
    # Save results
    try:
        fixation_output_path = os.path.join(processed_data_dir, f'fixation_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')
        saccade_output_path = os.path.join(processed_data_dir, f'saccade_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')
        with open(fixation_output_path, 'wb') as f:
            pickle.dump(fix_dict, f)
        logger.info(f"Fixation results saved successfully at {fixation_output_path}")
        with open(saccade_output_path, 'wb') as f:
            pickle.dump(sacc_dict, f)
        logger.info(f"Saccade results saved successfully at {saccade_output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Incorrect number of arguments. Usage: python script.py <task_key> <params_file_path>")
        sys.exit(1)
    task_key = sys.argv[1]
    params_file_path = sys.argv[2]
    main(task_key, params_file_path)
