import os
import logging
import pandas as pd
import numpy as np
import ssm
from tqdm import tqdm
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
    fit_arhmm_models(fix_binary_vector_df, params, n_states=3)
    pdb.set_trace()




def fit_arhmm_models(fix_binary_vector_df, params, n_states=3):
    """
    Fits ARHMM models for each unique m1-m2 pair and saves the models and predicted states.
    """
    # Ensure output directories exist
    os.makedirs(params['ssm_models_dir'], exist_ok=True)
    os.makedirs(params['processed_data_dir'], exist_ok=True)
    
    # Initialize dataframe to store predicted state sequences
    predicted_states_list = []
    
    # Group data by unique (m1, m2) pairs
    for (m1, m2), group_df in fix_binary_vector_df.groupby(['m1', 'm2']):
        
        # Dictionary to hold models nested under fixation type
        arhmm_models = {}
        
        for fixation_type in ['eyes', 'face']:
            
            # Extract data for fixation type
            fix_type_df = group_df[group_df['fixation_type'] == fixation_type]
            
            # Further group by session, interaction type, and run number
            session_groups = fix_type_df.groupby(['session_name', 'interaction_type', 'run_number'])
            
            # Collect data across all runs for model fitting
            m1_data = []
            m2_data = []
            stacked_data = []
            
            for _, session_df in session_groups:
                m1_rows = session_df[session_df['agent'] == 'm1']['binary_vector'].tolist()
                m2_rows = session_df[session_df['agent'] == 'm2']['binary_vector'].tolist()
                
                if not m1_rows or not m2_rows:
                    continue
                
                m1_data.extend(m1_rows)
                m2_data.extend(m2_rows)
                stacked_data.extend([np.stack((m1_v, m2_v), axis=0) for m1_v, m2_v in zip(m1_rows, m2_rows)])
            
            if not m1_data or not m2_data:
                continue  # Skip if no valid data
            
            # Initialize ARHMMs
            arhmm_models.setdefault(fixation_type, {})
            arhmm_models[fixation_type]['m1'] = ssm.HMM(n_states, obs_dim=1)
            arhmm_models[fixation_type]['m2'] = ssm.HMM(n_states, obs_dim=1)
            arhmm_models[fixation_type]['m1_m2'] = ssm.HMM(n_states, obs_dim=2)
            
            # Fit models across all runs for the m1-m2 pair
            arhmm_models[fixation_type]['m1'].fit(m1_data, method='em', num_iters=50)
            arhmm_models[fixation_type]['m2'].fit(m2_data, method='em', num_iters=50)
            arhmm_models[fixation_type]['m1_m2'].fit(stacked_data, method='em', num_iters=50)
        
        # Save models after both fixation types are processed
        model_path = os.path.join(params['ssm_models_dir'], f"{m1}_{m2}_arhmm.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(arhmm_models, f)
        
        # Predict hidden states
        for fixation_type, models in arhmm_models.items():
            for _, session_df in group_df[group_df['fixation_type'] == fixation_type].groupby(['session_name', 'interaction_type', 'run_number']):
                for i, row in session_df.iterrows():
                    agent = row['agent']
                    data = np.array(row['binary_vector']).reshape(-1, 1)
                    pred_states = models[agent].most_likely_states(data)
                    
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
                    
                    # Predict hidden states for m1_m2 model
                    if agent == 'm1':
                        paired_data = np.stack((data.flatten(), np.array(session_df[session_df['agent'] == 'm2']['binary_vector'].iloc[0]).flatten()), axis=0)
                        pred_states_m1_m2 = models['m1_m2'].most_likely_states(paired_data.T)
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
    
    # Create dataframe of predicted states
    predicted_states_df = pd.DataFrame(predicted_states_list)
    
    # Save to file
    output_path = os.path.join(params['processed_data_dir'], 'predicted_states.pkl')
    predicted_states_df.to_pickle(output_path)



# ** Call to main() **

if __name__ == "__main__":
    main()