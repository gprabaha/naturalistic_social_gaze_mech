import numpy as np
import pandas as pd
import os
import psutil
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import pickle
import glob
import warnings
from joblib import Parallel, delayed

import random
import ssm  # Import Switching State-Space Models
from ssm.util import find_permutation
from ssm.extensions.mp_srslds.transitions_ext import StickyRecurrentOnlyTransitions, StickyRecurrentTransitions

import pdb

import load_data
import curate_data
from hpc_slds_fitter import HPCSLDSFitter


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
        'is_cluster': False,
        'is_grace': False,
        'num_slds_states': 3,
        'num_slds_latents': 2,
        'num_slds_iters': 10,
        'remake_fixation_timeline': True,
        'run_locally': True,
        'fit_slds_for_agents_in_serial': False,
        'shuffle_task_order': True,
        'test_single_task': False  # Set to True to test a single random task
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

    pdb.set_trace()
    
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
    """Returns the number of allocated CPUs and dynamically adjusts threads per CPU based on SLURM settings."""
    # Get number of CPUs allocated by SLURM
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    slurm_cpus = int(slurm_cpus) if slurm_cpus else 1  # Default to 1 if not in SLURM
    # Default to 4 threads per CPU unless num_cpus is less than 4
    threads_per_cpu = 3 if slurm_cpus >= 3 else 1
    # Compute num_cpus by dividing total CPUs by threads per CPU
    num_cpus = max(1, slurm_cpus // threads_per_cpu)  # Ensure at least 1 CPU
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
        "out_of_roi": 3#,
        #'object': 4
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
    Submits SLDS fitting tasks to the HPC using job arrays or runs them locally in serial,
    waits for completion, and then loads and concatenates the results into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing all SLDS fitting results.
    """
    logger.info("Preparing SLDS fitting tasks.")

    grouped = fixation_timeline_df.groupby(["session_name", "interaction_type", "run_number"])
    task_keys = [f"{session},{interaction_type},{run}" for (session, interaction_type, run), _ in grouped]

    if params.get("shuffle_task_order", False):
        np.random.shuffle(task_keys)
    
    if params.get("test_single_task", False):
        task_keys = [random.choice(task_keys)]
        logger.info(f"Running a single test task: {task_keys[0]}")

    logger.info(f"Generated {len(task_keys)} SLDS tasks.")

    if params.get("run_locally", False):
        logger.info("Running SLDS fitting tasks locally in serial mode.")
        results_list = []
        for task_key in tqdm(task_keys, desc="Processing SLDS tasks"):
            session_name, interaction_type, run_number = task_key.split(',')
            run_number = int(run_number)
            group_df = fixation_timeline_df[
                (fixation_timeline_df['session_name'] == session_name) &
                (fixation_timeline_df['interaction_type'] == interaction_type) &
                (fixation_timeline_df['run_number'] == run_number)
            ]
            try:
                result = fit_slds_to_timeline_pair(group_df, params)
                results_list.append(result)
            except Exception as e:
                logger.error(f"Error processing {task_key}: {e}")

    else:
        logger.info("Submitting SLDS fitting tasks to HPC.")
        hpc_fitter = HPCSLDSFitter(params)
        params_file_path = os.path.join(params['processed_data_dir'], "params.pkl")
        with open(params_file_path, 'wb') as f:
            pickle.dump(params, f)

        job_file_path = hpc_fitter.generate_job_file(task_keys, params_file_path)
        hpc_fitter.submit_job_array(job_file_path)

        # Collect and process SLDS results
        logger.info("Collecting SLDS results from HPC.")
        results_dir = params["slds_results_dir"]
        result_files = glob.glob(os.path.join(results_dir, "slds_*.pkl"))
        results_list = []
        for file in result_files:
            try:
                with open(file, "rb") as f:
                    result = pickle.load(f)
                    results_list.append(result)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    if not results_list:
        logger.warning("No SLDS results were found. Returning an empty DataFrame.")
        return pd.DataFrame()

    slds_results_df = pd.DataFrame(results_list)
    logger.info(f"Successfully loaded {len(slds_results_df)} SLDS results.")
    
    return slds_results_df


def fit_slds_to_timeline_pair(df, params):
    """
    Fit an SLDS model separately for each agent (m1, m2) and jointly for both.
    Stores ELBO, AIC, BIC, Latent States, Transition Matrices, and Emission Parameters.

    Args:
        df (pd.DataFrame): A dataframe containing fixation timelines.
        params (dict): Configuration parameters including parallelization settings.

    Returns:
        dict: Minimal SLDS output with transition matrices and emission parameters.
    """
    try:
        session_name = df["session_name"].iloc[0]
        interaction_type = df["interaction_type"].iloc[0]
        run_number = df["run_number"].iloc[0]

        logger.info(f"Starting SLDS fitting for session: {session_name}, interaction_type: {interaction_type}, run: {run_number}")

        timeline_m1 = np.asarray(df[df["agent"] == "m1"]["fixation_timeline"].values[0]).reshape(-1, 1)
        timeline_m2 = np.asarray(df[df["agent"] == "m2"]["fixation_timeline"].values[0]).reshape(-1, 1)

        T = timeline_m1.shape[0]
        num_states = params.get('num_slds_states', 2)
        latent_dim = params.get('num_slds_latents', 2)
        num_iters = params.get('num_slds_iters', 25)

        num_states_joint = num_states
        latent_dim_joint = latent_dim


        # One-hot encode fixation timelines
        timeline_m1_onehot = one_hot_encode_timeline(timeline_m1).astype(int)
        timeline_m2_onehot = one_hot_encode_timeline(timeline_m2).astype(int)

        obs_dim_m1 = timeline_m1_onehot.shape[1]
        obs_dim_m2 = timeline_m2_onehot.shape[1]
        timeline_joint_onehot = np.hstack((timeline_m1_onehot, timeline_m2_onehot))
        obs_dim_joint = timeline_joint_onehot.shape[1]

        if (params.get("run_locally", False) & params.get("fit_slds_for_agents_in_serial", False)):
            logger.info("Running SLDS fitting in serial mode.")
            results = [
                fit_slds(obs_dim_m1, timeline_m1_onehot, "m1", num_states, latent_dim, num_iters),
                fit_slds(obs_dim_m2, timeline_m2_onehot, "m2", num_states, latent_dim, num_iters),
                fit_slds(obs_dim_joint, timeline_joint_onehot, "joint", num_states_joint, latent_dim_joint, num_iters)
            ]
        else:
            logger.info("Running SLDS fitting in parallel.")
            results = Parallel(n_jobs=-1)(
                delayed(fit_slds)(obs_dim, timeline, label, discrete_states, latents, num_iters)
                for obs_dim, timeline, label, discrete_states, latents, num_iters in [
                    (obs_dim_m1, timeline_m1_onehot, "m1", num_states, latent_dim, num_iters),
                    (obs_dim_m2, timeline_m2_onehot, "m2", num_states, latent_dim, num_iters),
                    (obs_dim_joint, timeline_joint_onehot, "joint", num_states_joint, latent_dim_joint, num_iters)
                ]
            )
        # Extract ELBOs
        elbo_m1, elbo_m2, elbo_joint = results[0]["ELBO"], results[1]["ELBO"], results[2]["ELBO"]

        # Compute model complexity (K) for each model
        K_m1 = compute_K(obs_dim_m1, num_states, latent_dim)
        K_m2 = compute_K(obs_dim_m2, num_states, latent_dim)
        K_joint = compute_K(obs_dim_joint, num_states_joint, latent_dim_joint)

        # Compute AIC and BIC
        aic_m1, bic_m1 = compute_aic_bic(elbo_m1, T, K_m1)
        aic_m2, bic_m2 = compute_aic_bic(elbo_m2, T, K_m2)
        aic_joint, bic_joint = compute_aic_bic(elbo_joint, T, K_joint)

        return {
            "session_name": session_name,
            "interaction_type": interaction_type,
            "run_number": run_number,

            # Results for m1
            "ELBO_m1": elbo_m1,
            "AIC_m1": aic_m1,
            "BIC_m1": bic_m1,
            "Latent_States_m1": results[0]["Latent_States"],
            "Smoothed_Latents_m1": results[0]["Smoothed_Latents"],

            # Results for m2
            "ELBO_m2": elbo_m2,
            "AIC_m2": aic_m2,
            "BIC_m2": bic_m2,
            "Latent_States_m2": results[1]["Latent_States"],
            "Smoothed_Latents_m2": results[1]["Smoothed_Latents"],

            # Results for joint model
            "ELBO_joint": elbo_joint,
            "AIC_joint": aic_joint,
            "BIC_joint": bic_joint,
            "Latent_States_joint": results[2]["Latent_States"],
            "Smoothed_Latents_joint": results[2]["Smoothed_Latents"]
        }

    except Exception as e:
        logger.error(f"Error during SLDS fitting: {e}", exc_info=True)
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


def fit_slds(obs_dim, onehot_data, label, num_states=2, latent_dim=2, num_iters=25):
    """
    Fit an SLDS model for a given observation dimension and one-hot encoded data.
    
    Args:
        obs_dim (int): Observation dimension.
        onehot_data (np.ndarray): One-hot encoded input data.
        label (str): Identifier for logging.
        num_states (int): Number of discrete latent states.
        latent_dim (int): Number of continuous latent dimensions.
        num_iters (int): Number of iterations for fitting.

    Returns:
        dict: Contains ELBO, Smoothed Latents, Most Likely Discrete States, Transition Matrices, and Emission Parameters.
    """
    try:
        logger.info(f"Fitting SLDS model for {label} with obs_dim={obs_dim}, num_states={num_states}, latent_dim={latent_dim}")


        # Initialize the SLDS model
        slds = ssm.SLDS(
            obs_dim,
            num_states,
            latent_dim,
            emissions="bernoulli",
            transitions="recurrent_only"
        )
        slds.initialize(onehot_data, inputs=None, num_init_iters=100, num_init_restarts=5, verbose=1)

        logger.info(f"Starting SLDS fitting for {label}. Data shape: {onehot_data.shape}")

        # Fit the SLDS using BBVI and mean-field variational inference
        elbos, posterior = slds.fit(
            onehot_data, num_iters=num_iters, initialize=False,
            method="laplace_em", # "bbvi",
            variational_posterior="structured_meanfield", #, "meanfield",
            alpha = 0.5
            # stepsize=1000
        )
        
        elbo = elbos[-1]
        logger.info(f"SLDS fitting completed for {label}. Final ELBO: {elbo:.2f}")
        
        # Extract smoothed latent variables
        smoothed_latents = posterior.mean[0][1] # index 0 is the discrete posterior expectations
        logger.info(f"Extracted smoothed latent states for {label}. Shape: {smoothed_latents.shape}")
        
        # Extract most likely discrete latent states
        latent_states = slds.most_likely_states(smoothed_latents, onehot_data)
        logger.info(f"Extracted most likely discrete states for {label}. Length: {len(latent_states)}")
        # ** (This part is causing issues; many instances are going into infinite loops) Align discrete states using permutation if needed

        # perm = find_permutation(latent_states, slds.most_likely_states(smoothed_latents, onehot_data))
        # while len(set(perm)) < slds.K:
        #     perm = find_permutation(latent_states, slds.most_likely_states(smoothed_latents, onehot_data))
        # slds.permute(perm)
        # latent_states = slds.most_likely_states(smoothed_latents, onehot_data)
        
        return {
            "ELBO": elbo,
            "Smoothed_Latents": smoothed_latents,
            "Latent_States": latent_states
        }

    except Exception as e:
        logger.error(f"Error fitting SLDS for {label}: {e}", exc_info=True)
        return {
            "ELBO": -np.inf,
            "Smoothed_Latents": [],
            "Latent_States": []
        }


def compute_K(obs_dim, num_states, latent_dim):
    """
    Compute number of parameters (K) for the SLDS model.

    Args:
        obs_dim (int): Observation dimension.
        num_states (int): Number of discrete states.
        latent_dim (int): Number of continuous latent dimensions.

    Returns:
        int: Number of model parameters.
    """
    # 1. Emission parameters: Maps latent space to observations
    # - Each observed variable depends on latent variables (obs_dim * latent_dim)
    # - Plus a bias term per observed variable (+obs_dim)
    K_emission = (obs_dim * latent_dim) + obs_dim

    # 2. Latent dynamics parameters:
    # - Each discrete state has a transition matrix for latents (num_states * latent_dim^2)
    # - Plus a bias per latent variable (num_states * latent_dim)
    K_latent = (num_states * latent_dim * latent_dim) + (num_states * latent_dim)

    # 3. Switching probabilities:
    # - Transition matrix for discrete states (num_states * num_states), minus one row to enforce sum-to-one constraint
    K_switching = (num_states * num_states) - num_states

    # 4. Initial state distribution:
    # - Categorical distribution for discrete states (num_states - 1)
    # - Mean and covariance of latent states ((latent_dim * latent_dim) + latent_dim)
    K_initial = (num_states - 1) + (latent_dim * latent_dim) + latent_dim

    # Total number of parameters
    return K_emission + K_latent + K_switching + K_initial


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
