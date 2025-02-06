import sys
import pickle
import pandas as pd
from fit_slds_to_fix_timeline import fit_slds_to_timeline_pair

# Load input arguments
task_key = sys.argv[1]  # Example format: "session_name,interaction_type,run_number"
params_file_path = sys.argv[2]

# Load parameters (if needed)
with open(params_file_path, 'rb') as f:
    params = pickle.load(f)

# Extract session, interaction type, and run number from task key
session_name, interaction_type, run_number = task_key.split(',')

# Load dataset
data_path = params['fixation_timeline_df_path']
fixation_timeline_df = pd.read_pickle(data_path)

# Filter dataframe for the given session, interaction type, and run
group_df = fixation_timeline_df[
    (fixation_timeline_df['session_name'] == session_name) &
    (fixation_timeline_df['interaction_type'] == interaction_type) &
    (fixation_timeline_df['run_number'] == int(run_number))
]

# Fit SLDS
results = fit_slds_to_timeline_pair(group_df)

# Save results
output_dir = params['slds_results_dir']
output_path = f"{output_dir}/slds_{session_name}_{interaction_type}_{run_number}.pkl"

with open(output_path, 'wb') as f:
    pickle.dump(results, f)

print(f"Saved SLDS results to {output_path}")
