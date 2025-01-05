
import os

import curate_data
import load_data
import util

import pdb

# Initialize params with data paths
params = {}
params = curate_data.add_root_data_to_params(params)
params = curate_data.add_processed_data_to_params(params)
params = curate_data.add_raw_data_dir_to_params(params)
params = curate_data.add_paths_to_all_data_files_to_params(params)
params = curate_data.prune_data_file_paths_with_pos_time_filename_mismatch(params)


# Load synchronized m1 and m2 gaze data
synchronized_gaze_data_file_path = os.path.join(params['processed_data_dir'], 'synchronized_gaze_data_df.pkl')
synchronized_gaze_data_df = load_data.get_synchronized_gaze_data_df(synchronized_gaze_data_file_path)

# Check if there are any m1-m2 data length mismatches in teh synchronized gaze data df
mismatch_df = util.find_mismatched_gaze_data(synchronized_gaze_data_df)

# Display mismatch cases if any
if not mismatch_df.empty:
    print("Mismatch cases found:")
    display(mismatch_df)
else:
    print("No mismatches found. All groups passed.")

# Separate out sessions and run to process separately
df_keys_for_tasks = synchronized_gaze_data_df[['session_name', 'interaction_type', 'run_number', 'agent', 'positions']].values.tolist()

pdb.set_trace()

print(0)