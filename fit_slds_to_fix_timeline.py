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

    num_cpus, threads_per_cpu = get_slurm_cpus_and_threads()

    processed_data_dir = params['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)  # Ensure the directory exists

    eye_mvm_behav_df_file_path = os.path.join(processed_data_dir, 'eye_mvm_behav_df.pkl')
    monkeys_per_session_dict_file_path = os.path.join(processed_data_dir, 'ephys_days_and_monkeys.pkl')

    logger.info("Loading data files")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    monkeys_per_session_df = pd.DataFrame(load_data.get_data_df(monkeys_per_session_dict_file_path))
    logger.info("Data loaded successfully")

    # Merge monkey names into behavioral data
    eye_mvm_behav_df = eye_mvm_behav_df.merge(monkeys_per_session_df, on="session_name", how="left")

    # Generate fixation timeline
    fixation_timeline_df = generate_fixation_timeline(eye_mvm_behav_df)

    # Fit SLDS models for all cases
    fixation_timeline_slds_results_df = generate_slds_models(fixation_timeline_df)
    fixation_timeline_slds_results_df = fixation_timeline_slds_results_df.merge(monkeys_per_session_df, on="session_name", how="left")

    # Save results
    fixation_timeline_slds_results_df.to_pickle(os.path.join(processed_data_dir, 'fixation_timeline_slds.pkl'))

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
    Fit SLDS models for m1, m2, and joint cases for all session-run combinations.
    Uses tqdm for progress tracking.
    Returns a dataframe with ELBO, AIC, BIC, and latent states.
    """
    logger.info("Fitting SLDS models for all sessions and runs")
    
    tqdm.pandas()
    grouped_results = fixation_timeline_df.groupby(["session_name", "interaction_type", "run_number"]) \
                                          .progress_apply(fit_slds_to_timeline_pair)

    logger.info("SLDS fitting complete. Converting results to dataframe.")
    
    # Convert results to a DataFrame and reset index to retain group identifiers
    slds_results_df = pd.DataFrame(grouped_results.tolist(), index=grouped_results.index).reset_index()
    
    return slds_results_df

def fit_slds_to_timeline_pair(df):
    """
    Fit an SLDS model separately for each agent (m1, m2) and jointly for both.
    Compare ELBOs, compute AIC and BIC.
    
    Args:
        df (pd.DataFrame): A dataframe containing fixation timelines for both agents
                           in the same session, interaction type, and run.

    Returns:
        dict: A dictionary with ELBOs, AIC, BIC, and inferred latent states.
    """
    timeline_m1 = np.asarray(df[df["agent"] == "m1"]["fixation_timeline"].values[0]).reshape(-1, 1)
    timeline_m2 = np.asarray(df[df["agent"] == "m2"]["fixation_timeline"].values[0]).reshape(-1, 1)
    
    T = timeline_m1.shape[0]  # Number of time points

    if len(np.unique(timeline_m1)) < 2 or np.isnan(timeline_m1).any():
        return {"ELBO_m1": -np.inf, "ELBO_m2": -np.inf, "ELBO_joint": -np.inf}

    if len(np.unique(timeline_m2)) < 2 or np.isnan(timeline_m2).any():
        return {"ELBO_m1": -np.inf, "ELBO_m2": -np.inf, "ELBO_joint": -np.inf}

    num_states = 2
    latent_dim = 1

    # Fit SLDS for m1
    obs_dim = 1
    slds_m1 = ssm.SLDS(num_states, latent_dim, obs_dim, emissions="categorical", transitions="recurrent_only")
    slds_m1.initialize([timeline_m1])
    q_elbos_m1, _ = slds_m1.fit([timeline_m1], num_iters=50)
    elbo_m1 = q_elbos_m1[-1]  # NO NORMALIZATION

    # Fit SLDS for m2
    obs_dim = 1
    slds_m2 = ssm.SLDS(num_states, latent_dim, obs_dim, emissions="categorical", transitions="recurrent_only")
    slds_m2.initialize([timeline_m2])
    q_elbos_m2, _ = slds_m2.fit([timeline_m2], num_iters=50)
    elbo_m2 = q_elbos_m2[-1]  # NO NORMALIZATION

    # Fit SLDS jointly (obs_dim = 2)
    obs_dim = 2
    timeline_joint = np.hstack((timeline_m1, timeline_m2))  # Shape (T, 2)
    slds_joint = ssm.SLDS(num_states, latent_dim, obs_dim, emissions="categorical", transitions="recurrent_only")
    slds_joint.initialize([timeline_joint])
    q_elbos_joint, _ = slds_joint.fit([timeline_joint], num_iters=50)
    elbo_joint = q_elbos_joint[-1]  # NO NORMALIZATION

    # Compute AIC and BIC
    K_individual = num_states * latent_dim  # Parameters for single-agent model
    K_joint = num_states * latent_dim * 2  # Parameters for joint model

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
        "Latent_m1": slds_m1.most_likely_states([timeline_m1]),
        "Latent_m2": slds_m2.most_likely_states([timeline_m2]),
        "Latent_joint": slds_joint.most_likely_states([timeline_joint]),
    }

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
