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
        'refit_arhmm': False,
        'remake_fixation_timeline': True,
        'run_locally': True,
        'fit_slds_for_agents_in_serial': False,
        'shuffle_task_order': True,
        'test_single_task': False  # Set to True to test a single random task
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
    """
    Fits ARHMM models for each unique m1-m2 pair using Dynamax and saves the models.
    If `params['refit_arhmm'] == False`, loads precomputed models instead of refitting.
    """
    # Ensure output directories exist
    os.makedirs(params['ssm_models_dir'], exist_ok=True)

    # Step 1: Fit and Save Models (Only if refit is True)
    if params.get('refit_arhmm', True):
        logger.info("Refitting ARHMM models...")

        # Dictionary to store all models
        all_arhmm_models = {}

        for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
            arhmm_models = {}  # Stores models for this m1-m2 pair
            key = jr.PRNGKey(1)  # PRNG for randomness
            
            for fixation_type in ['eyes', 'face']:
                fix_type_df = group_df[group_df['fixation_type'] == fixation_type]
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
                    continue  # Skip if no valid data

                # Pad sequences
                m1_data_padded = pad_sequences(m1_data)
                m2_data_padded = pad_sequences(m2_data)

                assert m1_data_padded.shape == m2_data_padded.shape, \
                    f"Shape mismatch after padding: m1 {m1_data_padded.shape}, m2 {m2_data_padded.shape}"

                stacked_data_padded = jnp.stack((m1_data_padded, m2_data_padded), axis=-1)

                # Split PRNG key
                key_m1, key_m2, key_m1_m2 = jr.split(key, 3)

                # Initialize models
                arhmm_models.setdefault(fixation_type, {})
                arhmm_models[fixation_type]['m1'] = BernoulliHMM(n_states, 1)
                arhmm_models[fixation_type]['m2'] = BernoulliHMM(n_states, 1)
                arhmm_models[fixation_type]['m1_m2'] = BernoulliHMM(n_states, 2)

                # Fit models
                params_m1, props_m1 = arhmm_models[fixation_type]['m1'].initialize(key_m1, method="prior")
                params_m1, _ = arhmm_models[fixation_type]['m1'].fit_em(
                    params_m1, props_m1, jnp.expand_dims(m1_data_padded, axis=-1)
                )

                params_m2, props_m2 = arhmm_models[fixation_type]['m2'].initialize(key_m2, method="prior")
                params_m2, _ = arhmm_models[fixation_type]['m2'].fit_em(
                    params_m2, props_m2, jnp.expand_dims(m2_data_padded, axis=-1)
                )

                params_m1_m2, props_m1_m2 = arhmm_models[fixation_type]['m1_m2'].initialize(key_m1_m2, method="prior")
                params_m1_m2, _ = arhmm_models[fixation_type]['m1_m2'].fit_em(
                    params_m1_m2, props_m1_m2, stacked_data_padded
                )

                # Store parameters
                arhmm_models[fixation_type]['m1_params'] = params_m1
                arhmm_models[fixation_type]['m2_params'] = params_m2
                arhmm_models[fixation_type]['m1_m2_params'] = params_m1_m2

            # Save models
            model_path = os.path.join(params['ssm_models_dir'], f"{m1}_{m2}_bernoulli_hmm.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(arhmm_models, f)
            
            logger.info(f"Model saved to {model_path}")

            all_arhmm_models[(m1, m2)] = arhmm_models

    else:
        logger.info("Loading precomputed ARHMM models...")
        all_arhmm_models = {}

        for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
            model_path = os.path.join(params['ssm_models_dir'], f"{m1}_{m2}_bernoulli_hmm.pkl")

            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    arhmm_models = pickle.load(f)
                all_arhmm_models[(m1, m2)] = arhmm_models
            else:
                logger.info(f"Model file not found: {model_path}. Skipping.")

    # Step 2: Predict Hidden States
    predicted_states_list = []

    for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
        if (m1, m2) not in all_arhmm_models:
            continue  # Skip missing models

        arhmm_models = all_arhmm_models[(m1, m2)]

        for fixation_type, models in arhmm_models.items():
            for _, session_df in group_df[group_df['fixation_type'] == fixation_type].groupby(
                    ['session_name', 'interaction_type', 'run_number']):
                
                for i, row in session_df.iterrows():
                    agent = row['agent']
                    data = row['binary_vector']
                    data = jnp.array(data)[:, None]
                    pred_states = arhmm_models[fixation_type][agent].most_likely_states(
                        arhmm_models[fixation_type][f"{agent}_params"], jnp.expand_dims(data, axis=-1)
                    )

                    predicted_states_list.append({
                        'session_name': row['session_name'],
                        'interaction_type': row['interaction_type'],
                        'run_number': row['run_number'],
                        'm1': m1,
                        'm2': m2,
                        'agent': agent,
                        'fixation_type': fixation_type,
                        'predicted_states': pred_states.tolist()
                    })

                    if agent == 'm1':
                        paired_data = jnp.stack(
                            (row['binary_vector'], session_df[session_df['agent'] == 'm2']['binary_vector'].iloc[0]),
                            axis=1)
                        pred_states_m1_m2 = arhmm_models[fixation_type]['m1_m2'].most_likely_states(
                            arhmm_models[fixation_type]['m1_m2_params'], paired_data
                        )
                        predicted_states_list.append({
                            'session_name': row['session_name'],
                            'interaction_type': row['interaction_type'],
                            'run_number': row['run_number'],
                            'm1': m1,
                            'm2': m2,
                            'agent': 'm1_m2',
                            'fixation_type': fixation_type,
                            'predicted_states': pred_states_m1_m2.tolist()
                        })

    # Save predicted states
    predicted_states_df = pd.DataFrame(predicted_states_list)
    output_path = os.path.join(params['processed_data_dir'], 'bernoulli_predicted_states.pkl')
    predicted_states_df.to_pickle(output_path)

    logger.info("BernoulliHMM predictions saved.")


def pad_sequences(sequences, pad_value=0):
    """Pad variable-length sequences to the maximum length."""
    max_length = max(len(seq) for seq in sequences)
    
    # Pad each sequence to the same length
    padded_sequences = [jnp.pad(jnp.array(seq), (0, max_length - len(seq)), mode="constant", constant_values=pad_value)
                        for seq in sequences]
    
    return jnp.stack(padded_sequences)  # Ensures uniform shape


# ** Call to main() **

if __name__ == "__main__":
    main()