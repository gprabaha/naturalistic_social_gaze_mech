
import os
import random
import pandas as pd
import pickle

import curate_data
import load_data
import util
import hpc_fix_and_saccade_detector

import pdb

# Initialize params with data paths
params = {}
params.update({
    'try_using_single_run': True
    })
params = curate_data.add_root_data_to_params(params)
params = curate_data.add_processed_data_to_params(params)
params = curate_data.add_raw_data_dir_to_params(params)
params = curate_data.add_paths_to_all_data_files_to_params(params)
params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(params)

# Load synchronized m1 and m2 gaze data
synchronized_gaze_data_file_path = os.path.join(params['processed_data_dir'], 'synchronized_gaze_data_df.pkl')
synchronized_gaze_data_df = load_data.get_synchronized_gaze_data_df(synchronized_gaze_data_file_path)

# Separate out sessions and run to process separately
df_keys_for_tasks = synchronized_gaze_data_df[['session_name', 'interaction_type', 'run_number', 'agent', 'positions']].values.tolist()

use_one_run = params.get('try_using_single_run')
if use_one_run:
    random_run = random.choice(df_keys_for_tasks)
    random_session, random_interaction_type, random_run_num, random_agent, _ = random_run
    df_keys_for_tasks = [random_run]
    print(f"!! Testing using positions data from a random single run: {random_session}, {random_interaction_type}, {random_run_num}, {random_agent}!!")

params_file_path = os.path.join(params['processed_data_dir'], 'params.pkl')
with open(params_file_path, 'wb') as f:
    pickle.dump(params, f)
print(f"Pickle dumped params to {params_file_path}")

fixation_rows = []
saccade_rows = []
if params.get('recompute_fix_and_saccades_through_hpc_jobs', False):
    if params.get('recompute_fix_and_saccades', False):
        # Use HPCFixAndSaccadeDetector to submit jobs for each task
        detector = hpc_fix_and_saccade_detector.HPCFixAndSaccadeDetector(params)
        job_file_path = detector.generate_job_file(df_keys_for_tasks, params_file_path)
        detector.submit_job_array(job_file_path)
    # Combine results after jobs have completed
    hpc_data_subfolder = params.get('hpc_job_output_subfolder', '')
    for task in df_keys_for_tasks:
        session, interaction_type, run, agent, _ = task
        print(f'Updating fix/saccade results for: {session}, {interaction_type}, {run}, {agent}')
        run_str = str(run)
        fix_path = os.path.join(
            params['processed_data_dir'],
            hpc_data_subfolder,
            f'fixation_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
        sacc_path = os.path.join(
            params['processed_data_dir'],
            hpc_data_subfolder,
            f'saccade_results_{session}_{interaction_type}_{run_str}_{agent}.pkl')
        if os.path.exists(fix_path):
            with open(fix_path, 'rb') as f:
                fix_indices = pickle.load(f)
                fixation_rows.append({
                    'session_name': session,
                    'interaction_type': interaction_type,
                    'run_number': run,
                    'agent': agent,
                    'fixation_start_stop': fix_indices
                })
        if os.path.exists(sacc_path):
            with open(sacc_path, 'rb') as f:
                sacc_indices = pickle.load(f)
                saccade_rows.append({
                    'session_name': session,
                    'interaction_type': interaction_type,
                    'run_number': run,
                    'agent': agent,
                    'saccade_start_stop': sacc_indices
                })
else:
    for task in df_keys_for_tasks:
        session, interaction_type, run, agent, positions = task

# Convert fixation and saccade lists to DataFrames
fixation_df = pd.DataFrame(fixation_rows)
saccade_df = pd.DataFrame(saccade_rows)
print("Detection completed for both agents.")



def process_fix_and_saccade_for_specific_run(session_name, positions, params):

    return 0

print(0)