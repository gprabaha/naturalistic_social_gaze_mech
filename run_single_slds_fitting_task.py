import sys
import pickle
import pandas as pd
import logging
from fit_slds_to_fix_timeline import fit_slds_to_timeline_pair
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    # Log script start
    logger.info("Starting SLDS fitting task.")

    # Load input arguments
    if len(sys.argv) < 3:
        raise ValueError("Missing required arguments: task_key and params_file_path.")

    task_key = sys.argv[1]  # Example format: "session_name,interaction_type,run_number"
    params_file_path = sys.argv[2]

    logger.info(f"Received task: {task_key}")
    logger.info(f"Loading parameters from: {params_file_path}")

    # Load parameters
    with open(params_file_path, 'rb') as f:
        params = pickle.load(f)

    # Extract session, interaction type, and run number from task key
    session_name, interaction_type, run_number = task_key.split(',')

    logger.info(f"Processing session: {session_name}, interaction_type: {interaction_type}, run_number: {run_number}")

    # Load dataset
    data_path = params.get('fixation_timeline_df_path')
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Fixation timeline file not found at: {data_path}")

    logger.info(f"Loading fixation timeline data from: {data_path}")
    fixation_timeline_df = pd.read_pickle(data_path)

    # Filter dataframe for the given session, interaction type, and run
    logger.info("Filtering data for the current task.")
    group_df = fixation_timeline_df[
        (fixation_timeline_df['session_name'] == session_name) &
        (fixation_timeline_df['interaction_type'] == interaction_type) &
        (fixation_timeline_df['run_number'] == int(run_number))
    ]

    if group_df.empty:
        logger.warning(f"No data found for session {session_name}, interaction_type {interaction_type}, run_number {run_number}. Skipping task.")
        sys.exit(0)

    # Fit SLDS
    logger.info("Fitting SLDS model.")
    results = fit_slds_to_timeline_pair(group_df, params)

    # Save results
    output_dir = params.get('slds_results_dir')
    if not output_dir:
        raise ValueError("slds_results_dir is not defined in params.")

    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_path = os.path.join(output_dir, f"slds_{session_name}_{interaction_type}_{run_number}.pkl")

    logger.info(f"Saving SLDS results to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info("SLDS fitting task completed successfully.")

except Exception as e:
    logger.error(f"Error occurred: {e}", exc_info=True)
    sys.exit(1)  # Exit with error status
