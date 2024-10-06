
import pandas as pd
import numpy as np


# Main function to create the scaled autocorrelations DataFrame
def compute_scaled_autocorrelations_for_behav_df(df):
    autocorr_df = pd.DataFrame()
    autocorr_df['session_name'] = df['session_name']
    autocorr_df['interaction_type'] = df['interaction_type']
    autocorr_df['run_number'] = df['run_number']
    autocorr_df['agent'] = df['agent']
    # Compute scaled autocorrelations for fixation and saccade vectors
    autocorr_df['fixation_scaled_autocorr'] = df['fixation_binary_vector'].apply(_scaled_autocorrelation_for_bin_vector)
    autocorr_df['saccade_scaled_autocorr'] = df['saccade_binary_vector'].apply(_scaled_autocorrelation_for_bin_vector)
    return autocorr_df


# Subfunction for calculating scaled autocorrelations
def _scaled_autocorrelation_for_bin_vector(binary_vector):
    n = len(binary_vector)
    autocorrs = []
    for lag in range(n):
        valid_length = n - lag
        dot_product = np.dot(binary_vector[:valid_length], binary_vector[lag:])
        autocorrs.append(dot_product / valid_length)
    return autocorrs