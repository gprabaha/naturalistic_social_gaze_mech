import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import os


# Main function with optional parallel processing and pickling
def compute_and_save_scaled_autocorrelations_for_behavior_df(df, params):
    """
    Computes the scaled autocorrelations for fixation and saccade binary vectors across all rows of the given DataFrame.
    Saves the resulting DataFrame to the processed data directory as a pickle file.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing binary vectors for fixation and saccade events.
    params (dict): Dictionary containing configuration, including 'num_cpus', 'use_parallel', and 'processed_data_dir'.
    Returns:
    pd.DataFrame: A new DataFrame containing the autocorrelations for fixation and saccade vectors.
    """
    num_cpus = params['num_cpus']
    use_parallel = params['use_parallel']
    processed_data_dir = params['processed_data_dir']
    # Ensure processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)
    if use_parallel:
        # Run the computation in parallel using joblib and tqdm for progress tracking
        results = Parallel(n_jobs=num_cpus)(
            delayed(_compute_autocorr_for_row)(row) for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Making autocorrelations for df rows (Parallel)")
        )
    else:
        # Run the computation in serial mode with progress bar
        results = [_compute_autocorr_for_row(row) for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Making autocorrelations for df rows (Serial)")
        ]
    # Create a new DataFrame to store the results
    autocorr_df = pd.DataFrame()
    autocorr_df['session_name'] = df['session_name']
    autocorr_df['interaction_type'] = df['interaction_type']
    autocorr_df['run_number'] = df['run_number']
    autocorr_df['agent'] = df['agent']
    # Store the autocorrelation results
    autocorr_df['fixation_scaled_autocorr'] = [r[0] for r in results]
    autocorr_df['saccade_scaled_autocorr'] = [r[1] for r in results]
    # Pickle the resulting DataFrame
    output_path = os.path.join(processed_data_dir, 'scaled_autocorrelations.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(autocorr_df, f)
    print(f"DataFrame saved to {output_path}")
    return autocorr_df


# Helper function to process a single row of the DataFrame
def _compute_autocorr_for_row(row):
    """
    Computes scaled autocorrelations for both fixation and saccade binary vectors for a given row.
    Parameters:
    row (pd.Series): A single row from the DataFrame.
    Returns:
    tuple: A tuple containing scaled autocorrelations for fixation and saccade binary vectors.
    """
    fixation_autocorr = _compute_scaled_autocorr(row['fixation_binary_vector'])
    saccade_autocorr = _compute_scaled_autocorr(row['saccade_binary_vector'])
    return fixation_autocorr, saccade_autocorr


# Subfunction to calculate scaled autocorrelations for a binary vector
def _compute_scaled_autocorr(binary_vector):
    """
    Computes the scaled autocorrelation for a given binary vector.
    Parameters:
    binary_vector (list): A binary vector representing either fixation or saccade events.
    Returns:
    list: A list of scaled autocorrelation values.
    """
    n = len(binary_vector)
    autocorrs = []
    for lag in range(n):
        valid_length = n - lag  # Number of overlapping elements after the shift
        dot_product = np.dot(binary_vector[:valid_length], binary_vector[lag:])  # Dot product of overlapping elements
        autocorrs.append(dot_product / valid_length)  # Scale by the valid length
    return autocorrs
