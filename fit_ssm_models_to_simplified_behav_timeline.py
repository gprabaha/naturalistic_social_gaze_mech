import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model import BernoulliHMM
from jax.numpy import pad

import pdb

import load_data
import curate_data


## ** JAX Warning Suppression **
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppresses TensorFlow/JAX logs
os.environ["JAX_LOG_LEVEL"] = "ERROR"
os.environ["XLA_FLAGS"] = "--xla_disable_slow_operation_warnings"



## ** Configure logging **
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
        'refit_arhmm': True
    }
    
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)

    # Set SLDS results directory
    params['ssm_models_dir'] = os.path.join(params['processed_data_dir'], 'ssm_model_fits')
    
    os.makedirs(params['ssm_models_dir'], exist_ok=True)

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
    fix_binary_vector_file = os.path.join(processed_data_dir, 'fix_binary_vector_df.pkl')

    logger.info("Loading data files")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    monkeys_per_session_df = pd.DataFrame(load_data.get_data_df(monkeys_per_session_dict_file_path))
    fix_binary_vector_df = load_data.get_data_df(fix_binary_vector_file)
    logger.info("Data loaded successfully")

    # Merge monkey names into behavioral data
    eye_mvm_behav_df = eye_mvm_behav_df.merge(monkeys_per_session_df, on="session_name", how="left")
    fix_binary_vector_df = fix_binary_vector_df.merge(monkeys_per_session_df, on="session_name", how="left")

    # Fit ARHMM models
    fit_bernoulli_hmm_models(fix_binary_vector_df, params, n_states=3)



def fit_bernoulli_hmm_models(fix_binary_vector_df, params, n_states=3):
    """Fits BernoulliHMM models for each unique m1-m2 pair using Dynamax and saves the models."""
    os.makedirs(params['ssm_models_dir'], exist_ok=True)

    if params.get('refit_arhmm', True):
        logger.info("Refitting BernoulliHMM models...")
        all_arhmm_models = {}

        for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
            key = jr.PRNGKey(1)
            models = invoke_model_dict(n_states, key)

            arhmm_models = {}
            for fixation_type in ['eyes', 'face']:
                fix_type_df = group_df[group_df['fixation_type'] == fixation_type]
                arhmm_models[fixation_type] = fit_model_for_fix_type(fix_type_df, models, m1, m2, fixation_type)
            
            if arhmm_models:
                model_path = os.path.join(params['ssm_models_dir'], f"{m1}_{m2}_bernoulli_hmm.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(arhmm_models, f)
                logger.info(f"Model saved to {model_path}")

                all_arhmm_models[(m1, m2)] = arhmm_models

    else:
        logger.info("Loading precomputed BernoulliHMM models...")
        all_arhmm_models = {}
        for (m1, m2), _ in fix_binary_vector_df.groupby(['m1', 'm2']):
            model_path = os.path.join(params['ssm_models_dir'], f"{m1}_{m2}_bernoulli_hmm.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    all_arhmm_models[(m1, m2)] = pickle.load(f)

    predicted_states_df = predict_hidden_states(fix_binary_vector_df, all_arhmm_models)
    predicted_states_df.to_pickle(os.path.join(params['processed_data_dir'], 'bernoulli_predicted_states.pkl'))
    logger.info("BernoulliHMM predictions saved.")

def invoke_model_dict(n_states, key):
    """Initializes BernoulliHMM models for different agents."""
    key_m1, key_m2, key_m1_m2 = jr.split(key, 3)
    return {
        'm1': (BernoulliHMM(n_states, 1), key_m1),
        'm2': (BernoulliHMM(n_states, 1), key_m2),
        'm1_m2': (BernoulliHMM(n_states, 2), key_m1_m2)
    }

def fit_model_for_fix_type(fix_type_df, models, m1, m2, fixation_type):
    """Processes a single fixation type, fits models, and computes metrics."""
    logger.info(f"Fitting models for {m1}-{m2} {fixation_type} fixations")
    
    session_groups = fix_type_df.groupby(['session_name', 'interaction_type', 'run_number'])
    m1_data, m2_data = [], []
    for _, session_df in session_groups:
        m1_rows = session_df[session_df['agent'] == 'm1']['binary_vector'].tolist()
        m2_rows = session_df[session_df['agent'] == 'm2']['binary_vector'].tolist()
        
        if not m1_rows or not m2_rows:
            continue
        m1_data.extend(m1_rows)
        m2_data.extend(m2_rows)

    if not m1_data or not m2_data:
        return None  # No valid data

    m1_data_padded = jnp.array(pad_sequences(m1_data))
    m2_data_padded = jnp.array(pad_sequences(m2_data))

    assert m1_data_padded.shape == m2_data_padded.shape, \
        f"Shape mismatch after padding: m1 {m1_data_padded.shape}, m2 {m2_data_padded.shape}"

    stacked_data_padded = jnp.stack((m1_data_padded, m2_data_padded), axis=-1)

    # Fit models and store parameters
    arhmm_models = {}
    for agent, (model, key) in models.items():
        arhmm_models[agent] = model
        data = jnp.expand_dims(m1_data_padded, axis=-1) if agent == 'm1' else jnp.expand_dims(m2_data_padded, axis=-1) if agent == 'm2' else stacked_data_padded

        params, props = model.initialize(key, method="prior")
        params, _ = model.fit_em(params, props, data)
        arhmm_models[f"{agent}_params"] = params

        # Compute and store model metrics
        metrics = compute_model_metrics(model, params, data)
        arhmm_models[f"{agent}_metrics"] = metrics
        logger.info(f"{agent}: Log-Likelihood={metrics['log_likelihood']:.2f}, AIC={metrics['AIC']:.2f}, BIC={metrics['BIC']:.2f}")

    return arhmm_models

def pad_sequences(sequences, pad_value=0):
    """Pad variable-length sequences to the maximum length."""
    max_length = max(len(seq) for seq in sequences)
    
    # Pad each sequence to the same length
    padded_sequences = [jnp.pad(jnp.array(seq), (0, max_length - len(seq)), mode="constant", constant_values=pad_value)
                        for seq in sequences]
    
    return jnp.stack(padded_sequences)  # Ensures uniform shape

def compute_model_metrics(model, params, data):
    """
    Computes log-likelihood, AIC, and BIC for a fitted BernoulliHMM.
    
    Args:
        model: The fitted BernoulliHMM model.
        params: The parameters of the fitted model.
        data: The observed data (array-like).

    Returns:
        A dictionary containing the log-likelihood, AIC, and BIC.
    """
    # Compute the log-likelihood of the data under the model
    log_likelihood = model.marginal_log_prob(params, data)
    
    # Number of states
    num_states = params.transitions.transition_matrix.shape[0]
    
    # Number of observation dimensions
    num_obs_dim = params.emissions.probs.shape[1]
    
    # Calculate the number of parameters
    num_params = (
        num_states * (num_states - 1) +  # Transition probabilities (excluding one due to sum-to-one constraint)
        num_states * num_obs_dim +       # Emission probabilities
        num_states                       # Initial state probabilities
    )
    
    # Number of observations
    T = data.shape[0]
    
    # Calculate AIC and BIC
    AIC = 2 * num_params - 2 * log_likelihood
    BIC = np.log(T) * num_params - 2 * log_likelihood
    
    return {'log_likelihood': log_likelihood, 'AIC': AIC, 'BIC': BIC}

def predict_hidden_states(fix_binary_vector_df, all_arhmm_models):
    """Predicts hidden states using fitted HMM models."""
    predicted_states_list = []
    
    for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
        if (m1, m2) not in all_arhmm_models:
            continue
        
        logger.info(f"Making state predictions for {m1}-{m2}")
        arhmm_models = all_arhmm_models[(m1, m2)]

        for fixation_type, models in arhmm_models.items():
            for _, session_df in group_df[group_df['fixation_type'] == fixation_type].groupby(
                    ['session_name', 'interaction_type', 'run_number']):
                
                for _, row in session_df.iterrows():
                    agent = row['agent']
                    data = jnp.array(row['binary_vector'])[:, None]

                    pred_states = arhmm_models[fixation_type][agent].most_likely_states(
                        arhmm_models[fixation_type][f"{agent}_params"], data
                    )

                    predicted_states_list.append({
                        'session_name': row['session_name'],
                        'interaction_type': row['interaction_type'],
                        'run_number': row['run_number'],
                        'm1': m1, 'm2': m2,
                        'agent': agent, 'fixation_type': fixation_type,
                        'predicted_states': pred_states.tolist()
                    })

                    if agent == 'm1':
                        paired_data = jnp.stack(
                            (row['binary_vector'], session_df[session_df['agent'] == 'm2']['binary_vector'].iloc[0]),
                            axis=1
                        )
                        pred_states_m1_m2 = arhmm_models[fixation_type]['m1_m2'].most_likely_states(
                            arhmm_models[fixation_type]['m1_m2_params'], paired_data
                        )
                        predicted_states_list.append({
                            'session_name': row['session_name'],
                            'interaction_type': row['interaction_type'],
                            'run_number': row['run_number'],
                            'm1': m1, 'm2': m2,
                            'agent': 'm1_m2', 'fixation_type': fixation_type,
                            'predicted_states': pred_states_m1_m2.tolist()
                        })

    return pd.DataFrame(predicted_states_list)



# ** Call to main() **

if __name__ == "__main__":
    main()