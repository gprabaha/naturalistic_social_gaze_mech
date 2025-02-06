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
import glob
import warnings
from joblib import Parallel, delayed

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
        'remake_fixation_timeline': True,
        'test_single_task': True  # Set to True to test a single random task
    }
    
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)

    # Define processed data directory
    processed_data_dir = params['processed_data_dir']
    
    # Set SLDS results directory
    params['slds_results_dir'] = os.path.join(processed_data_dir, 'slds_results')
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
    if params.get('remake_fixation_timeline', False):
        logger.info("Generating fixation timeline")
        fixation_timeline_df = generate_fixation_timeline(eye_mvm_behav_df)
        fixation_timeline_df.to_pickle(fixation_timeline_path)
    else:
        logger.info("Loading existing fixation timeline")
        fixation_timeline_df = pd.read_pickle(fixation_timeline_path)

    # Submit SLDS fitting jobs and retrieve results
    fixation_timeline_slds_results_df = generate_slds_models(fixation_timeline_df, params)

    # Merge SLDS results with monkey session data
    if not fixation_timeline_slds_results_df.empty:
        fixation_timeline_slds_results_df = fixation_timeline_slds_results_df.merge(
            monkeys_per_session_df, on="session_name", how="left"
        )

    # Save results
    final_output_path = os.path.join(processed_data_dir, 'fixation_timeline_slds.pkl')
    fixation_timeline_slds_results_df.to_pickle(final_output_path)
    logger.info(f"Saved final SLDS results to {final_output_path}")



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
    Submits SLDS fitting tasks to the HPC using job arrays, waits for completion, 
    and then loads and concatenates the results into a single DataFrame.
    
    If `params['test_single_task']` is True, runs a single random task instead of all.
    
    Returns:
        pd.DataFrame: A DataFrame containing all SLDS fitting results.
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

    # Wait for jobs to complete and collect results
    logger.info("Collecting SLDS results.")
    results_dir = params["slds_results_dir"]
    result_files = glob.glob(os.path.join(results_dir, "slds_*.pkl"))

    results_list = []
    for file in result_files:
        try:
            with open(file, "rb") as f:
                result = pickle.load(f)
                # Extract session details from the filename
                filename = os.path.basename(file)
                parts = filename.replace("slds_", "").replace(".pkl", "").split("_")
                session_name, interaction_type, run_number = parts[0], parts[1], int(parts[2])
                
                # Add grouping columns
                result["session_name"] = session_name
                result["interaction_type"] = interaction_type
                result["run_number"] = run_number

                results_list.append(result)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not results_list:
        logger.warning("No SLDS results were found. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Convert results into a DataFrame
    slds_results_df = pd.DataFrame(results_list)

    logger.info(f"Successfully loaded {len(slds_results_df)} SLDS results.")
    
    return slds_results_df


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
    try:
        session_name = df["session_name"].iloc[0]
        interaction_type = df["interaction_type"].iloc[0]
        run_number = df["run_number"].iloc[0]
        logger.info("Starting SLDS fitting for session: %s, interaction_type: %s, run: %s",
                    session_name, interaction_type, run_number)

        timeline_m1 = np.asarray(df[df["agent"] == "m1"]["fixation_timeline"].values[0]).reshape(-1, 1)
        timeline_m2 = np.asarray(df[df["agent"] == "m2"]["fixation_timeline"].values[0]).reshape(-1, 1)

        T = timeline_m1.shape[0]  # Number of time points
        logger.info("Number of time points in fixation timeline: %d", T)

        # Check if the timeline contains only one unique state or NaN values
        if len(np.unique(timeline_m1)) < 2 or np.isnan(timeline_m1).any():
            logger.warning("Skipping SLDS fitting: m1 timeline contains insufficient variance or NaN values.")
            return {"ELBO_m1": -np.inf, "ELBO_m2": -np.inf, "ELBO_joint": -np.inf}

        if len(np.unique(timeline_m2)) < 2 or np.isnan(timeline_m2).any():
            logger.warning("Skipping SLDS fitting: m2 timeline contains insufficient variance or NaN values.")
            return {"ELBO_m1": -np.inf, "ELBO_m2": -np.inf, "ELBO_joint": -np.inf}

        num_states = 2
        latent_dim = 1

        # One-hot encode fixation timelines
        timeline_m1_onehot = one_hot_encode_timeline(timeline_m1)
        timeline_m2_onehot = one_hot_encode_timeline(timeline_m2)
        logger.info("One-hot encoding completed. Shapes - m1: %s, m2: %s", timeline_m1_onehot.shape, timeline_m2_onehot.shape)

        obs_dim_m1 = timeline_m1_onehot.shape[1]
        obs_dim_m2 = timeline_m2_onehot.shape[1]

        # Prepare joint data
        timeline_joint_onehot = np.hstack((timeline_m1_onehot, timeline_m2_onehot))
        obs_dim_joint = timeline_joint_onehot.shape[1]

        # Fit SLDS for m1, m2, and joint in parallel
        results = Parallel(n_jobs=3)(
            delayed(fit_slds)(obs_dim_m1, timeline_m1_onehot, "m1"),
            delayed(fit_slds)(obs_dim_m2, timeline_m2_onehot, "m2"),
            delayed(fit_slds)(obs_dim_joint, timeline_joint_onehot, "joint")
        )

        (elbo_m1, latent_m1), (elbo_m2, latent_m2), (elbo_joint, latent_joint) = results

        # Compute model parameters
        K_individual = (num_states * latent_dim) + (latent_dim * obs_dim_m1)
        K_joint = (num_states * latent_dim) + (latent_dim * obs_dim_joint)

        aic_m1, bic_m1 = compute_aic_bic(elbo_m1, T, K_individual)
        aic_m2, bic_m2 = compute_aic_bic(elbo_m2, T, K_individual)
        aic_joint, bic_joint = compute_aic_bic(elbo_joint, 2 * T, K_joint)

        logger.info("SLDS fitting completed successfully.")

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
            "Latent_m1": latent_m1,
            "Latent_m2": latent_m2,
            "Latent_joint": latent_joint,
        }

    except Exception as e:
        logger.error("Error during SLDS fitting: %s", e, exc_info=True)
        return {}


def one_hot_encode_timeline(timeline):
    """
    Convert categorical timeline into a one-hot encoded matrix.
    Handles cases where some categories might be missing in the sequence.

    Args:
        timeline (np.ndarray): A 1D array of categorical observations.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (T, num_categories).
    """
    try:
        num_categories = len(np.unique(timeline))
        logger.info("One-hot encoding timeline with %d categories.", num_categories)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            encoder = OneHotEncoder(sparse=False, categories=[list(range(num_categories))])
            encoded_timeline = encoder.fit_transform(timeline.reshape(-1, 1))
            encoded_timeline = encoded_timeline.astype(int)

        logger.info("One-hot encoding successful. Output shape: %s", encoded_timeline.shape)
        return encoded_timeline

    except Exception as e:
        logger.error("Error during one-hot encoding: %s", e, exc_info=True)
        return np.zeros((timeline.shape[0], 1))  # Return a zero matrix to avoid crashes


def fit_slds(obs_dim, onehot_data, label, num_states=2, latent_dim=1):
    """
    Fit an SLDS model for a given observation dimension and one-hot encoded data.
    
    Args:
        obs_dim (int): Observation dimension.
        onehot_data (np.ndarray): One-hot encoded input data.
        label (str): Identifier for logging.
        num_states (int): Number of discrete latent states.
        latent_dim (int): Number of continuous latent dimensions.

    Returns:
        tuple: (ELBO, Latent states)
    """
    try:
        logger.info(f"Fitting SLDS model for {label}.")
        slds = ssm.SLDS(obs_dim, num_states, latent_dim, emissions="bernoulli", transitions="recurrent_only")
        slds.inputs = None
        slds.initialize([onehot_data], inputs=None)
        q_elbos, _ = slds.fit([onehot_data], num_iters=50)
        elbo = q_elbos[-1]
        latent_states = slds.most_likely_states([onehot_data])
        return elbo, latent_states
    except Exception as e:
        logger.error(f"Error fitting SLDS for {label}: {e}", exc_info=True)
        return -np.inf, []


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
    try:
        aic = -2 * elbo + 2 * K
        bic = -2 * elbo + K * np.log(T)
        logger.info("Computed AIC: %.2f, BIC: %.2f", aic, bic)
        return aic, bic

    except Exception as e:
        logger.error("Error computing AIC/BIC: %s", e, exc_info=True)
        return np.nan, np.nan



# ** Call to main() **



if __name__ == "__main__":
    main()
