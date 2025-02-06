import numpy as np
import pandas as pd
import os
import psutil
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import ssm  # Import Switching State-Space Models

import pdb

import load_data
import curate_data

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# ** Initiation and Main **

def _initialize_params():
    logger.info("Initializing parameters")
    params = {
        'is_cluster': True,
        'is_grace': False
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    logger.info("Parameters initialized successfully")
    return params

def main():
    logger.info("Starting the script")
    params = _initialize_params()

    processed_data_dir = params['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)  # Ensure the directory exists

    eye_mvm_behav_df_file_path = os.path.join(processed_data_dir, 'eye_mvm_behav_df.pkl')
    monkeys_per_session_dict_file_path = os.path.join(processed_data_dir, 'ephys_days_and_monkeys.pkl')

    logger.info("Loading data files")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    monkeys_per_session_df = pd.DataFrame(load_data.get_data_df(monkeys_per_session_dict_file_path))
    logger.info("Data loaded successfully")

    # Generate fixation timeline
    fixation_timeline_df = generate_fixation_timeline(eye_mvm_behav_df)

    # Fit SLDS models
    fixation_timeline_df = generate_slds_models(fixation_timeline_df)

    # Save results
    fixation_timeline_df.to_pickle(os.path.join(processed_data_dir, 'fixation_timeline_slds.pkl'))
    
    num_cpus, threads_per_cpu = get_slurm_cpus_and_threads()
    pdb.set_trace()


# ** Sub-functions **

## Count available CPUs and threads
def get_slurm_cpus_and_threads():
    """Returns the number of allocated CPUs and threads per CPU using psutil."""
    
    # Get number of CPUs allocated by SLURM
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK"))

    # Get total virtual CPUs (logical cores)
    total_logical_cpus = psutil.cpu_count(logical=True)

    # Calculate threads per CPU
    threads_per_cpu = total_logical_cpus // num_cpus if num_cpus > 0 else 1

    return num_cpus, threads_per_cpu


def generate_fixation_timeline(eye_mvm_behav_df):
    """
    Generate a timeline vector for fixation categories across the entire run length.
    
    Args:
        eye_mvm_behav_df (pd.DataFrame): Dataframe containing fixation event data.

    Returns:
        pd.DataFrame: New dataframe with an additional column 'fixation_timeline' storing the generated vectors.
    """
    fix_df = eye_mvm_behav_df.copy()
    tqdm.pandas()  # Enable progress tracking
    fix_df["fixation_timeline"] = fix_df.progress_apply(create_timeline, axis=1)
    return fix_df

def create_timeline(row):
    """Generate a fixation timeline vector for a single row in the dataframe."""
    category_map = {
        "eyes": 1,
        "non_eye_face": 2,
        "out_of_roi": 3
    }
    
    timeline = np.zeros(row["run_length"], dtype=int)
    categorized_fixations = categorize_fixations(row["fixation_location"])
    
    for (start, stop), category in zip(row["fixation_start_stop"], categorized_fixations):
        timeline[start:stop + 1] = category_map.get(category, 0)  # Default to 0 if category not found
    
    return timeline

def categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]


# ** SLDS Functions **

def generate_slds_models(fixation_timeline_df):
    """
    Fit an SLDS model to the fixation timeline of each row.
    
    Args:
        fixation_timeline_df (pd.DataFrame): Dataframe containing fixation timeline vectors.

    Returns:
        pd.DataFrame: Updated dataframe with an additional column for SLDS latent states.
    """
    tqdm.pandas()
    fixation_timeline_df["slds_latent_states"] = fixation_timeline_df.progress_apply(fit_slds_to_timeline, axis=1)
    return fixation_timeline_df

def fit_slds_to_timeline(row):
    """
    Fit an SLDS model to a single row's fixation timeline.

    Args:
        row (pd.Series): A row from the dataframe containing 'fixation_timeline'.

    Returns:
        np.ndarray: The inferred latent states from the SLDS model.
    """
    timeline = row["fixation_timeline"]
    
    if len(np.unique(timeline)) < 2:
        return np.zeros_like(timeline)  # No transitions if only one state is present
    
    num_states = 3  # Three fixation states (eyes, non_eye_face, out_of_roi)
    obs_dim = 1  # Observed dimension is just the timeline sequence
    
    # Define SLDS model
    slds = ssm.SLDS(num_states, obs_dim, transitions="recurrent_only")
    
    # Fit SLDS model
    slds.initialize(timeline.reshape(-1, 1))
    q_elbos, q_states = slds.fit(timeline.reshape(-1, 1), num_iters=50)
    
    # Get inferred latent states
    latent_states = slds.most_likely_states(timeline.reshape(-1, 1))
    
    return latent_states


# ** Call to main() **

if __name__ == "__main__":
    main()
