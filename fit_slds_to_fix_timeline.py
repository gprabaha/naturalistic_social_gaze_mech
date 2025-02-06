import numpy as np
import pandas as pd
import os
import psutil
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import random
import ssm  # Import Switching State-Space Models
from hpc_slds_fitter import HPCSLDSFitter
import pickle

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
    """
    Initializes parameters and adds root and processed data directories.
    """
    logger.info("Initializing parameters")
    
    params = {
        'is_cluster': True,
        'is_grace': False,
        'test_single_task': False  # Set to True to test a single random task
    }
    
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)

    # Define processed data directory
    processed_data_dir = params['processed_data_dir']
    
    # Set SLDS results directory
    params['slds_results_dir'] = os.path.join(processed_data_dir, params['slds_results_out_path'])
    os.makedirs(params['slds_results_dir'], exist_ok=True)

    # Path to the fixation timeline dataframe
    params['fixation_timeline_df_path'] = os.path.join(processed_data_dir, "fixation_timeline_df.pkl")

    logger.info("Parameters initialized successfully")
    return params


def main():
    """
    Main function to load data, generate fixation timeline, and submit SLDS fitting jobs.
    """
    logger.info("Starting the script")
    params = _initialize_params()

    processed_data_dir = params['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)  # Ensure the directory exists

    # Define paths
    eye_mvm_behav_df_file_path = os.path.join(processed_data_dir, 'eye_mvm_behav_df.pkl')
    monkeys_per_session_dict_file_path = os.path.join(processed_data_dir, 'ephys_days_and_monkeys.pkl')
    fixation_timeline_path = params['fixation_timeline_df_path']  # Now stored in params

    logger.info("Loading data files")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    monkeys_per_session_df = pd.DataFrame(load_data.get_data_df(monkeys_per_session_dict_file_path))
    logger.info("Data loaded successfully")

    # Merge monkey names into behavioral data
    eye_mvm_behav_df = eye_mvm_behav_df.merge(monkeys_per_session_df, on="session_name", how="left")

    # Generate fixation timeline if not already saved
    if not os.path.exists(fixation_timeline_path):
        logger.info("Generating fixation timeline")
        fixation_timeline_df = generate_fixation_timeline(eye_mvm_behav_df)
        fixation_timeline_df.to_pickle(fixation_timeline_path)
    else:
        logger.info("Loading existing fixation timeline")
        fixation_timeline_df = pd.read_pickle(fixation_timeline_path)

    # Submit SLDS fitting jobs
    generate_slds_models(fixation_timeline_df, params)


    pdb.set_trace()
    logger.info("SLDS model fitting complete. Results saved.")



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
        "out_of_roi": 3,
        'object': 4
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



def generate_slds_models(fixation_timeline_df, params):
    """
    Submits SLDS fitting tasks to the HPC using job arrays.
    
    If `params['test_single_task']` is True, runs a single random task instead of all.
    """
    logger.info("Preparing SLDS fitting tasks.")

    # Group data by session, interaction type, and run
    grouped = fixation_timeline_df.groupby(["session_name", "interaction_type", "run_number"])

    # Generate task keys
    task_keys = [f"{session},{interaction_type},{run}" for (session, interaction_type, run), _ in grouped]

    # Option to run a single random task for testing
    if params.get("test_single_task", False):
        task_keys = [random.choice(task_keys)]
        logger.info(f"Running a single test task: {task_keys[0]}")

    logger.info(f"Generated {len(task_keys)} SLDS tasks.")

    # Initialize HPC job submission class
    hpc_fitter = HPCSLDSFitter(params)
    
    # Save params to be used by individual job scripts
    params_file_path = os.path.join(params['processed_data_dir'], "params.pkl")
    with open(params_file_path, 'wb') as f:
        pickle.dump(params, f)

    # Generate and submit job file
    job_file_path = hpc_fitter.generate_job_file(task_keys, params_file_path)
    hpc_fitter.submit_job_array(job_file_path)

    logger.info("SLDS jobs submitted successfully.")


def fit_slds_to_timeline_pair(df):
    """
    Fit an SLDS model separately for each agent (m1, m2) and jointly for both.
    One-hot encodes categorical data before fitting.
    Computes ELBOs, AIC, and BIC.
    
    Args:
        df (pd.DataFrame): A dataframe containing fixation timelines for both agents
                           in the same session, interaction type, and run.

    Returns:
        dict: A dictionary with ELBOs, AIC, BIC, and inferred latent states.
    """
    timeline_m1 = np.asarray(df[df["agent"] == "m1"]["fixation_timeline"].values[0]).reshape(-1, 1)
    timeline_m2 = np.asarray(df[df["agent"] == "m2"]["fixation_timeline"].values[0]).reshape(-1, 1)

    T = timeline_m1.shape[0]  # Number of time points

    # Check if the timeline contains only one unique state or NaN values
    if len(np.unique(timeline_m1)) < 2 or np.isnan(timeline_m1).any():
        return {"ELBO_m1": -np.inf, "ELBO_m2": -np.inf, "ELBO_joint": -np.inf}

    if len(np.unique(timeline_m2)) < 2 or np.isnan(timeline_m2).any():
        return {"ELBO_m1": -np.inf, "ELBO_m2": -np.inf, "ELBO_joint": -np.inf}

    num_states = 2
    latent_dim = 1

    # One-hot encode fixation timelines
    timeline_m1_onehot = one_hot_encode_timeline(timeline_m1)
    timeline_m2_onehot = one_hot_encode_timeline(timeline_m2)
    timeline_m1_onehot = timeline_m1_onehot.astype(int)
    timeline_m2_onehot = timeline_m2_onehot.astype(int)

    # Dynamically determine the observation dimension after encoding
    obs_dim_m1 = timeline_m1_onehot.shape[1]
    obs_dim_m2 = timeline_m2_onehot.shape[1]
    
    # Fit SLDS for m1
    slds_m1 = ssm.SLDS(obs_dim_m1, num_states, latent_dim, emissions="bernoulli", transitions="recurrent_only")
    slds_m1.inputs = None
    slds_m1.initialize([timeline_m1_onehot], inputs=None)
    q_elbos_m1, _ = slds_m1.fit([timeline_m1_onehot], num_iters=50)
    elbo_m1 = q_elbos_m1[-1]  # NO NORMALIZATION

    # Fit SLDS for m2
    slds_m2 = ssm.SLDS(obs_dim_m2, num_states, latent_dim, emissions="bernoulli", transitions="recurrent_only")
    slds_m2.inputs = None
    slds_m2.initialize([timeline_m2_onehot], inputs=None)
    q_elbos_m2, _ = slds_m2.fit([timeline_m2_onehot], num_iters=50)
    elbo_m2 = q_elbos_m2[-1]  # NO NORMALIZATION

    # Fit SLDS jointly (obs_dim = sum of both one-hot encoded dimensions)
    timeline_joint_onehot = np.hstack((timeline_m1_onehot, timeline_m2_onehot))  # Shape (T, obs_dim_m1 + obs_dim_m2)
    obs_dim_joint = timeline_joint_onehot.shape[1]

    slds_joint = ssm.SLDS(obs_dim_joint, num_states, latent_dim, emissions="bernoulli", transitions="recurrent_only")
    slds_joint.inputs = None
    slds_joint.initialize([timeline_joint_onehot], inputs=None)
    q_elbos_joint, _ = slds_joint.fit([timeline_joint_onehot], num_iters=50)
    elbo_joint = q_elbos_joint[-1]  # NO NORMALIZATION

    # Compute number of model parameters correctly
    K_individual = (num_states * latent_dim) + (latent_dim * obs_dim_m1)  # Single-agent model
    K_joint = (num_states * latent_dim) + (latent_dim * obs_dim_joint)  # Joint model

    aic_m1, bic_m1 = compute_aic_bic(elbo_m1, T, K_individual)
    aic_m2, bic_m2 = compute_aic_bic(elbo_m2, T, K_individual)
    aic_joint, bic_joint = compute_aic_bic(elbo_joint, 2 * T, K_joint)

    return {
        "ELBO_m1": elbo_m1,
        "ELBO_m2": elbo_m2,
        "ELBO_joint": elbo_joint,
        "AIC_m1": aic_m1,
        "AIC_m2": aic_m2,
        "AIC_joint": aic_joint,
        "BIC_m1": bic_m1,
        "BIC_m2": bic_m2,
        "BIC_joint": bic_joint,
        "Latent_m1": slds_m1.most_likely_states([timeline_m1_onehot]),
        "Latent_m2": slds_m2.most_likely_states([timeline_m2_onehot]),
        "Latent_joint": slds_joint.most_likely_states([timeline_joint_onehot]),
    }

def one_hot_encode_timeline(timeline):
    """
    Convert categorical timeline into a one-hot encoded matrix.
    Handles cases where some categories might be missing in the sequence.
    
    Args:
        timeline (np.ndarray): A 1D array of categorical observations.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (T, num_categories).
    """
    num_categories = len(np.unique(timeline))  # Count unique categories in the data
    encoder = OneHotEncoder(sparse=False, categories=[list(range(num_categories))])
    return encoder.fit_transform(timeline.reshape(-1, 1))  # One-hot encode timeline

def compute_aic_bic(elbo, T, K):
    """
    Compute AIC and BIC scores.
    
    Args:
        elbo (float): ELBO value.
        T (int): Number of observations.
        K (int): Number of model parameters.
    
    Returns:
        tuple: (AIC, BIC)
    """
    aic = -2 * elbo + 2 * K
    bic = -2 * elbo + K * np.log(T)
    return aic, bic



# ** Call to main() **



if __name__ == "__main__":
    main()
