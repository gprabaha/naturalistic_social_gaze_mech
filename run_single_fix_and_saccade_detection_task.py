import pickle
import sys
import os
import pandas as pd

import load_data
import detect_eye_mvm_behav_from_gaze_data

def main(task_key, params_file_path):
    """
    Main function to run fixation and saccade detection for a single task.
    Parameters:
    - task_key (str): Key arguments to fetch the specific run from the gaze data dictionary.
    - params_file_path (str): Path to the serialized params file.
    """
    print("Starting the main function")
    # Load params to get the processed data directory path
    try:
        with open(params_file_path, 'rb') as f:
            params = pickle.load(f)
        print("Parameters loaded successfully.")
    except Exception as e:
        print(f"Failed to load parameters from {params_file_path}: {e}")
        sys.exit(1)
    # Paths to gaze data dictionary and missing data paths
    processed_data_dir = params['processed_data_dir']
    sparse_nan_removed_sync_gaze_data_df_path = os.path.join(processed_data_dir, 'sparse_nan_removed_sync_gaze_data_df.pkl')
    # Load gaze data dictionary
    try:
        sparse_nan_removed_sync_gaze_data_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_path)
        print("Gaze data dataframe loaded successfully.")
    except Exception as e:
        print(f"Failed to load gaze data from {sparse_nan_removed_sync_gaze_data_df_path}: {e}")
        sys.exit(1)
    # Parse task_key to fetch the required positions array
    try:
        # Parse the task key to get session, interaction type, run, and agent
        session, interaction_type, run, agent = task_key.split(',')
        session = session.strip()
        interaction_type = interaction_type.strip()
        run = int(run.strip())  # Ensure run is an integer
        agent = agent.strip()
        print(f"Parsed task key successfully: session: {session}; interaction_type: {interaction_type}; run: {str(run)}; agent: {agent}")
    except Exception as e:
        print(f"Failed to parse task key {task_key}: {e}")
        sys.exit(1)
    # Filter the DataFrame to get the matching row
    try:
        # Filter the DataFrame for the specific task
        filtered_df = sparse_nan_removed_sync_gaze_data_df[
            (sparse_nan_removed_sync_gaze_data_df['session_name'] == session) &
            (sparse_nan_removed_sync_gaze_data_df['interaction_type'] == interaction_type) &
            (sparse_nan_removed_sync_gaze_data_df['run_number'] == run) &
            (sparse_nan_removed_sync_gaze_data_df['agent'] == agent)
        ]
        if filtered_df.empty:
            print(f"No data found for task key: {task_key}")
            sys.exit(1)
        # Extract positions
        positions = filtered_df['positions'].iloc[0]  # Assuming 'positions' is a single column per row
        print("Successfully extracted positions for fixation and saccade detection.")
        # Run fixation and saccade detection
        fix_indices, sacc_indices, microsacc_indices = \
            detect_eye_mvm_behav_from_gaze_data._detect_fixations_saccades_and_microsaccades_in_run(positions, session)
        print("Fixation and saccade detection completed successfully.")
    except Exception as e:
        print(f"Failed during fixation and saccade detection: {e}")
        sys.exit(1)

    # Save results
    try:
        hpc_data_subfolder = params.get('hpc_job_output_subfolder', '')
        fixation_output_path = os.path.join(processed_data_dir, hpc_data_subfolder, f'fixation_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')
        saccade_output_path = os.path.join(processed_data_dir, hpc_data_subfolder, f'saccade_results_{session}_{interaction_type}_{str(run)}_{agent}.pkl')
        os.makedirs(os.path.dirname(fixation_output_path), exist_ok=True)
        with open(fixation_output_path, 'wb') as f:
            pickle.dump(fix_indices, f)
        print(f"Fixation results saved successfully at {fixation_output_path}")
        with open(saccade_output_path, 'wb') as f:
            pickle.dump((sacc_indices, microsacc_indices), f)
        print(f"Saccade results saved successfully at {saccade_output_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect number of arguments. Usage: python script.py <task_key> <params_file_path>")
        sys.exit(1)
    task_key = sys.argv[1]
    params_file_path = sys.argv[2]
    main(task_key, params_file_path)
