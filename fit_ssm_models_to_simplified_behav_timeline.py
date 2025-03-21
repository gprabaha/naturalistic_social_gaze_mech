import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

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
        'refit_hmm': True,
        'remake_predicted_states': True,
        'transition_matrix_stickiness': 0.9,
        'randomization_key_seed': 42,
        'num_indep_model_initiations': 10,
        'num_states_hmm': 3, # predicted social states: high, low, other
        'num_states_hmm_joint': 5, # predicted joint social states: high-high, high-low, low-high, low-low, other
        'make_hmm_state_plots': True
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
    fix_binary_vector_file_path = os.path.join(processed_data_dir, 'fix_binary_vector_df.pkl')

    logger.info("Loading data files...")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    monkeys_per_session_df = pd.DataFrame(load_data.get_data_df(monkeys_per_session_dict_file_path))
    fix_binary_vector_df = load_data.get_data_df(fix_binary_vector_file_path)
    logger.info("Data loaded successfully")

    # Merge monkey names into behavioral data
    eye_mvm_behav_df = eye_mvm_behav_df.merge(monkeys_per_session_df, on="session_name", how="left")
    fix_binary_vector_df = fix_binary_vector_df.merge(monkeys_per_session_df, on="session_name", how="left")

    predicted_states_file_path = os.path.join(params['processed_data_dir'], 'bernoulli_predicted_states.pkl')
    if params.get('remake_predicted_states', True):
        # Fit hmm models
        logger.info("Predicting states in timeline again...")
        predicted_states_df = fit_bernoulli_hmm_models(fix_binary_vector_df, params)
    else:
        logger.info("Loading previously indentified states dataframe...")
        predicted_states_df = load_data.get_data_df(predicted_states_file_path)
    
    if params.get('make_hmm_state_plots', True):
        logger.info("Making HMM state prediction plots...")
        plot_hmm_state_predictions(predicted_states_df, params)

    return 0


def fit_bernoulli_hmm_models(fix_binary_vector_df, params):
    """Fits BernoulliHMM models for each unique m1-m2 pair using Dynamax and saves the models."""
    os.makedirs(params['ssm_models_dir'], exist_ok=True)
    if params.get('refit_hmm', True):
        logger.info("Refitting BernoulliHMM models...")
        rand_key_seed = params.get('randomization_key_seed', 42)
        num_indep_inits = params.get('num_indep_model_initiations', 5)
        all_hmm_models = {}
        for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
            logger.info(f"Fitting models for {m1}-{m2}")
            key = jr.PRNGKey(rand_key_seed)
            models = invoke_model_dict(params, key)
            hmm_models = {}
            for fixation_type in ['eyes', 'face']:
                fix_type_df = group_df[group_df['fixation_type'] == fixation_type]
                hmm_models[fixation_type] = fit_model_for_fix_type(fix_type_df, models, m1, m2, fixation_type, num_indep_inits)
            if hmm_models:
                model_path = os.path.join(params['ssm_models_dir'], f"{m1}_{m2}_bernoulli_hmm.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(hmm_models, f)
                logger.info(f"Model saved to {model_path}")
                all_hmm_models[(m1, m2)] = hmm_models
    else:
        logger.info("Loading precomputed BernoulliHMM models...")
        all_hmm_models = load_precomputed_hmm_models(fix_binary_vector_df, params)
    predicted_states_df = predict_hidden_states(fix_binary_vector_df, all_hmm_models)
    predicted_states_df.to_pickle(os.path.join(params['processed_data_dir'], 'bernoulli_predicted_states.pkl'))
    logger.info("BernoulliHMM predictions saved.")
    return predicted_states_df


def invoke_model_dict(params, key):
    """Initializes BernoulliHMM models for different agents."""
    n_states = params.get('num_states_hmm', 0)
    n_states_joint = params.get('num_states_hmm_joint', 0)
    key_m1, key_m2, key_m1_m2 = jr.split(key, 3)
    transition_matrix_stickiness = params.get('transition_matrix_stickiness', 0.5)
    return {
        'm1': (
            BernoulliHMM(
                n_states,
                1,
                transition_matrix_stickiness=transition_matrix_stickiness
            ),
            key_m1),
        'm2': (
            BernoulliHMM(
                n_states,
                1,
                transition_matrix_stickiness=transition_matrix_stickiness
            ),
            key_m2),
        'm1_m2': (
            BernoulliHMM(
                n_states_joint,
                2,
                transition_matrix_stickiness=transition_matrix_stickiness
            ),
            key_m1_m2)
    }


def fit_model_for_fix_type(fix_type_df, models, m1, m2, fixation_type, num_indep_inits=5):
    """Processes a single fixation type, fits models multiple times in parallel, and selects the best fit."""
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
    # Convert lists to JAX arrays after padding
    m1_data_padded = jnp.expand_dims(jnp.array(pad_sequences(m1_data)), axis=-1)
    m2_data_padded = jnp.expand_dims(jnp.array(pad_sequences(m2_data)), axis=-1)
    assert m1_data_padded.shape == m2_data_padded.shape, \
        f"Shape mismatch after padding: m1 {m1_data_padded.shape}, m2 {m2_data_padded.shape}"
    stacked_data_padded = jnp.concatenate((m1_data_padded, m2_data_padded), axis=-1)
    # Determine number of parallel jobs
    num_cpus = min(int(os.getenv("SLURM_CPUS_PER_TASK", "1")), num_indep_inits)
    num_cpus = max(1, num_cpus)  # Ensure at least one process
    # Fit models in parallel and select the best fit
    hmm_models = {}
    for agent, (model, key) in models.items():
        logger.info(f"Fitting {agent} model for {m1}-{m2} {fixation_type} fixations with {num_indep_inits} initializations")
        data = m1_data_padded if agent == 'm1' else \
               m2_data_padded if agent == 'm2' else stacked_data_padded
        key_seeds = jr.split(key, num_indep_inits)
        # Parallel fitting with different initial conditions
        results = Parallel(n_jobs=num_cpus, backend="threading")(
            delayed(fit_model_once)(model, key_seeds[i], data) for i in range(num_indep_inits)
        )
        # Select the best fit based on the highest final log-likelihood
        best_params, best_log_likelihoods, best_metrics = max(results, key=lambda x: x[1][-1])
        hmm_models[agent] = model
        hmm_models[f"{agent}_params"] = best_params
        hmm_models[f"{agent}_log_likelihoods"] = best_log_likelihoods
        hmm_models[f"{agent}_metrics"] = best_metrics
        logger.info(f"{agent}: Log-Likelihood={best_metrics['log_likelihood']:.2f}, "
                    f"AIC={best_metrics['AIC']:.2f}, BIC={best_metrics['BIC']:.2f}")
    return hmm_models


def pad_sequences(sequences, pad_value=0):
    """Pad variable-length sequences to the maximum length."""
    max_length = max(len(seq) for seq in sequences)
    # Pad each sequence to the same length
    padded_sequences = [jnp.pad(jnp.array(seq), (0, max_length - len(seq)), mode="constant", constant_values=pad_value)
                        for seq in sequences]
    return jnp.stack(padded_sequences)  # Ensures uniform shape


def fit_model_once(model, key, data):
    """Fits a single model initialization and returns parameters, log-likelihoods, and metrics."""
    params, props = model.initialize(key, method="prior")
    params, log_likelihoods = model.fit_em(params, props, data, num_iters=100)
    metrics = compute_model_metrics(log_likelihoods[-1], params, data)
    return params, log_likelihoods, metrics


def compute_model_metrics(final_log_likelihood, params, data):
    """
    Computes log-likelihood, AIC, and BIC for a fitted BernoulliHMM.
    Args:
        final_log_likelihood: The final log-likelihood from fit_em.
        params: The parameters of the fitted model.
        data: The observed data (array-like).
    Returns:
        A dictionary containing the log-likelihood, AIC, and BIC.
    """
    assert data.ndim == 3, f"Expected (B, T, D), got {data.shape}"
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
    # Number of total observations across all runs
    T = max(data.shape[1], 1)  # T is the second dim
    B = data.shape[0]  # Number of runs
    # Compute AIC and BIC
    AIC = 2 * num_params - 2 * final_log_likelihood
    BIC = np.log(T * B) * num_params - 2 * final_log_likelihood  # Log total samples
    return {'log_likelihood': final_log_likelihood, 'AIC': AIC, 'BIC': BIC}


def load_precomputed_hmm_models(fix_binary_vector_df, params):
    """Loads precomputed HMM models from disk."""
    all_hmm_models = {}
    for (m1, m2), _ in fix_binary_vector_df.groupby(['m1', 'm2']):
        model_path = os.path.join(params['ssm_models_dir'], f"{m1}_{m2}_bernoulli_hmm.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                all_hmm_models[(m1, m2)] = pickle.load(f)
    return all_hmm_models


def predict_hidden_states(fix_binary_vector_df, all_hmm_models):
    """Predicts hidden states using fitted HMM models."""
    predicted_states_list = []
    for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
        if (m1, m2) not in all_hmm_models:
            continue
        logger.info(f"Making state predictions for {m1}-{m2}")
        hmm_models = all_hmm_models[(m1, m2)]
        for fixation_type, models in hmm_models.items():
            for (session, _, _), session_df in tqdm(
                group_df[group_df['fixation_type'] == fixation_type].groupby(
                    ['session_name', 'interaction_type', 'run_number']), 
                desc=f"{fixation_type} fixations"):
                for _, row in session_df.iterrows():
                    agent = row['agent']
                    data = jnp.array(row['binary_vector'])[:, None]
                    pred_states = hmm_models[fixation_type][agent].most_likely_states(
                        hmm_models[fixation_type][f"{agent}_params"], data
                    )
                    predicted_states_list.append({
                        'session_name': row['session_name'],
                        'interaction_type': row['interaction_type'],
                        'run_number': row['run_number'],
                        'm1': m1, 'm2': m2,
                        'timeline_of': agent, 'fixation_type': fixation_type,
                        'observed_timeline': data.tolist(),
                        'predicted_states': pred_states.tolist()
                    })
                    if agent == 'm1':
                        paired_data = jnp.stack(
                            (row['binary_vector'], session_df[session_df['agent'] == 'm2']['binary_vector'].iloc[0]),
                            axis=1
                        )
                        pred_states_m1_m2 = hmm_models[fixation_type]['m1_m2'].most_likely_states(
                            hmm_models[fixation_type]['m1_m2_params'], paired_data
                        )
                        predicted_states_list.append({
                            'session_name': row['session_name'],
                            'interaction_type': row['interaction_type'],
                            'run_number': row['run_number'],
                            'm1': m1, 'm2': m2,
                            'timeline_of': 'm1_m2', 'fixation_type': fixation_type,
                            'observed_timeline': paired_data.tolist(),
                            'predicted_states': pred_states_m1_m2.tolist()
                        })
    return pd.DataFrame(predicted_states_list)



def plot_hmm_state_predictions(predicted_states_df, params):
    # Create root plot directory
    root_plot_dir = os.path.join(params['root_data_dir'], 'plots', 'hmm_state_predictions')
    date_dir = datetime.today().strftime('%Y-%m-%d')
    plot_dir = os.path.join(root_plot_dir, date_dir)
    os.makedirs(plot_dir, exist_ok=True)
    # Parallel processing
    Parallel(n_jobs=-1)(delayed(generate_plot)(m1, m2, session_name, run_number, run_df, plot_dir)
                        for (m1, m2, session_name, run_number), run_df in tqdm(predicted_states_df.groupby(['m1', 'm2', 'session_name', 'run_number']), desc='Generating plots'))



def generate_plot(m1, m2, session_name, run_number, run_df, plot_dir):
    pair_dir = os.path.join(plot_dir, f"{m1}_{m2}")
    os.makedirs(pair_dir, exist_ok=True)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 6), sharex=True, sharey=True)
    for (fixation_type, timeline_of), row_df in run_df.groupby(['fixation_type', 'timeline_of']):
        row = ['m1', 'm2', 'm1_m2'].index(timeline_of)
        col = ['eyes', 'face'].index(fixation_type)
        if row_df.empty:
            axes[row, col].set_visible(False)
            continue
        observed_timeline = np.array(row_df.iloc[0]['observed_timeline'])
        predicted_states = np.array(row_df.iloc[0]['predicted_states'])
        unique_states = np.unique(predicted_states)
        time_axis = np.arange(len(predicted_states)) / 1000  # Convert to seconds
        for state in unique_states:
            mask = predicted_states == state
            axes[row, col].fill_between(time_axis, -0.5, 1.5, where=mask, color=f'C{state}', alpha=0.4)
        if timeline_of == 'm1_m2':
            axes[row, col].step(time_axis, observed_timeline[:, 0], color='blue', alpha=0.6)
            axes[row, col].step(time_axis, observed_timeline[:, 1], color='red', alpha=0.6)
        else:
            axes[row, col].step(time_axis, observed_timeline, color='blue', alpha=0.7)
        axes[row, col].set_title(f"{timeline_of.upper()} - {fixation_type.capitalize()}")
        axes[row, col].set_ylim(-0.5, 1.5)
        axes[row, col].set_xlabel("Time (seconds)")
        axes[row, col].set_ylabel("State / Observed Timeline")
    # Create a single legend for all subplots
    handles = [plt.Line2D([0], [0], color=f'C{i}', lw=4, label=f'State {i}') for i in range(5)]
    handles.append(plt.Line2D([0], [0], color='blue', lw=2, label='M1 Observed'))
    handles.append(plt.Line2D([0], [0], color='red', lw=2, label='M2 Observed'))
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Session {session_name}, Run {run_number} - {m1} & {m2} HMM State Predictions")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    # Save plot
    plot_filename = f"session-{session_name}_run-{run_number}_hmm_state_predictions.png"
    plot_path = os.path.join(pair_dir, plot_filename)
    fig.savefig(plot_path)
    plt.close(fig)

# ** Call to main() **

if __name__ == "__main__":
    main()