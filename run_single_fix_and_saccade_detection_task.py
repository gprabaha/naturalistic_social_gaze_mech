import pickle
import sys
import os
from util import get_gaze_data_dict
from fix_and_saccades import process_fix_and_saccade_for_specific_run

def main(task_key, params_file_path):
    """
    Main function to run fixation and saccade detection for a single task.
    Parameters:
    - task_key (str): Key arguments to fetch the specific run from the gaze data dictionary.
    - params_file_path (str): Path to the serialized params file.
    - agent (str): The agent ('m1' or 'm2') for which to run the detection.
    """
    # Load params to get the processed data directory path
    with open(params_file_path, 'rb') as f:
        params = pickle.load(f)

    # Paths to gaze data dictionary and missing data paths
    processed_data_dir = params['processed_data_dir']
    gaze_data_file_path = os.path.join(processed_data_dir, 'gaze_data_dict.pkl')
    missing_data_file_path = os.path.join(processed_data_dir, 'missing_data_dict_paths.pkl')

    # Load gaze data dictionary
    gaze_data_dict, _ = get_gaze_data_dict(gaze_data_file_path, missing_data_file_path)

    # Parse task_key to fetch the required positions array
    session, interaction_type, run, agent = task_key.split(',')
    session = session.strip()
    interaction_type = interaction_type.strip()
    run = int(run.strip())
    agent = agent.strip()

    # Prepare task arguments
    task_args = (session, interaction_type, run, agent, params)

    # Run fixation and saccade detection
    fix_dict, sacc_dict = process_fix_and_saccade_for_specific_run(task_args, gaze_data_dict)

    # Save results
    fixation_output_path = os.path.join(processed_data_dir, f'fixation_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')
    saccade_output_path = os.path.join(processed_data_dir, f'saccade_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')

    with open(fixation_output_path, 'wb') as f:
        pickle.dump(fix_dict, f)
    with open(saccade_output_path, 'wb') as f:
        pickle.dump(sacc_dict, f)

if __name__ == "__main__":
    task_key = sys.argv[1]
    params_file_path = sys.argv[2]
    main(task_key, params_file_path)
