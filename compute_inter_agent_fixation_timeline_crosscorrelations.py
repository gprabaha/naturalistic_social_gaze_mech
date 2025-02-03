import numpy as np
import pandas as pd
import os
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from multiprocessing import Pool
from tqdm import tqdm

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
        'reanalyse_fixation_probabilities': False,
        'reanalyze_fixation_crosscorrelations': False
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    logger.info("Parameters initialized successfully")
    return params


def main():
    logger.info("Starting the script")
    params = _initialize_params()

    eye_mvm_behav_df_file_path = os.path.join(
        params['processed_data_dir'], 'eye_mvm_behav_df.pkl'
    )
    monkeys_per_session_dict_file_path = os.path.join(
        params['processed_data_dir'], 'ephys_days_and_monkeys.pkl'
    )

    logger.info("Loading data files")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    monkeys_per_session_df = pd.DataFrame(load_data.get_data_df(monkeys_per_session_dict_file_path))
    logger.info("Data loaded successfully")
    pdb.set_trace()
    logger.info("Generating fix-related binary vectors")
    fix_binary_vector_df = _generate_fixation_binary_vectors(eye_mvm_behav_df)
    logger.info("Fix-related binary vectors generated")

    _compute_crosscorr_and_shuffled_stats(
        fix_binary_vector_df, eye_mvm_behav_df, sigma=25, num_shuffles=20, num_cpus=8
    )


# ** Sub-functions **


def _generate_fixation_binary_vectors(df):
    """
    Generate a long-format dataframe with binary vectors for each fixation category.

    Args:
        df (pd.DataFrame): DataFrame with fixation start-stop indices and fixation locations.

    Returns:
        pd.DataFrame: A long-format DataFrame where each row represents a fixation type's binary vector.
    """
    binary_vectors = []

    for _, row in tqdm(df.iterrows(), desc="Processing df row"):
        run_length = row["run_length"]
        fixation_start_stop = row["fixation_start_stop"]
        fixation_categories = __categorize_fixations(row["fixation_location"])

        binary_dict = {
            "eyes": np.zeros(run_length, dtype=int),
            "face": np.zeros(run_length, dtype=int),
            "out_of_roi": np.zeros(run_length, dtype=int),
        }

        for (start, stop), category in zip(fixation_start_stop, fixation_categories):
            if category == "eyes":
                binary_dict["eyes"][start:stop + 1] = 1
                binary_dict["face"][start:stop + 1] = 1  # 'face' includes 'eyes'
            elif category == "non_eye_face":
                binary_dict["face"][start:stop + 1] = 1
            elif category == "out_of_roi":
                binary_dict["out_of_roi"][start:stop + 1] = 1

        for fixation_type, binary_vector in binary_dict.items():
            binary_vectors.append({
                "session_name": row["session_name"],
                "interaction_type": row["interaction_type"],
                "run_number": row["run_number"],
                "agent": row["agent"],
                "fixation_type": fixation_type,
                "binary_vector": binary_vector
            })

    return pd.DataFrame(binary_vectors)

def __categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]


import numpy as np
import pandas as pd
import os
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from multiprocessing import Pool
from tqdm import tqdm
import load_data
import curate_data

def __generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, num_shuffles=20):
    """
    Generate shuffled versions of a binary fixation vector by randomly redistributing fixation intervals.
    """
    row = eye_mvm_behav_df[(eye_mvm_behav_df["session_name"] == session) &
                            (eye_mvm_behav_df["interaction_type"] == interaction) &
                            (eye_mvm_behav_df["run_number"] == run)].iloc[0]
    
    run_length = row["run_length"]
    categories = __categorize_fixations(row["fixation_location"])
    fixation_start_stop = row["fixation_start_stop"]
    
    # Get only the fixations of the requested type
    fixation_intervals = [
        (start, stop) for (start, stop), category in zip(fixation_start_stop, categories) if category == fixation_type
    ]
    fixation_durations = [stop - start + 1 for start, stop in fixation_intervals]
    total_fixation_duration = sum(fixation_durations)
    N = len(fixation_durations)
    
    # Compute total available non-fixation duration
    available_non_fixation_duration = run_length - total_fixation_duration
    
    shuffled_vectors = []
    for _ in range(num_shuffles):
        # Generate N+1 random non-fixation partitions that sum to available duration
        non_fixation_durations = np.random.multinomial(
            available_non_fixation_duration, 
            np.ones(N + 1) / (N + 1)
        )
        
        # Label fixation and non-fixation segments to ensure correct placement
        fixation_labels = [(dur, 1) for dur in fixation_durations]
        non_fixation_labels = [(dur, 0) for dur in non_fixation_durations]
        all_segments = fixation_labels + non_fixation_labels
        np.random.shuffle(all_segments)
        
        # Construct the shuffled binary vector
        shuffled_vector = np.zeros(run_length, dtype=int)
        index = 0
        for duration, label in all_segments:
            if index >= run_length:
                logging.warning("Index exceeded run length during shuffle generation")
                break
            if label == 1:
                shuffled_vector[index: index + duration] = 1  # Fixation interval
            index += duration
        
        shuffled_vectors.append(shuffled_vector)
    
    return shuffled_vectors


def _compute_crosscorr_and_shuffled_stats(df, eye_mvm_behav_df, sigma=25, num_shuffles=20, num_cpus=8):
    """
    Compute Gaussian-smoothed cross-correlation between m1 and m2's fixation behaviors.
    """
    results = []
    grouped = df.groupby(["session_name", "interaction_type", "run_number", "fixation_type"])
    
    for (session, interaction, run, fixation_type), group in grouped:
        if len(group) != 2:
            continue  # Skip if both m1 and m2 aren't present
        
        m1_row = group[group["agent"] == "m1"].iloc[0]
        m2_row = group[group["agent"] == "m2"].iloc[0]

        m1_vector = np.array(m1_row["binary_vector"])
        m2_vector = np.array(m2_row["binary_vector"])
        
        # Apply Gaussian smoothing
        m1_smooth = gaussian_filter1d(m1_vector, sigma)
        m2_smooth = gaussian_filter1d(m2_vector, sigma)
        
        # Compute original cross-correlations
        crosscorr_m1_m2, crosscorr_m2_m1 = __fft_crosscorrelation_both(m1_smooth, m2_smooth)
        
        # Generate shuffled vectors
        m1_shuffled_vectors = __generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, num_shuffles)
        m2_shuffled_vectors = __generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, num_shuffles)
        
        # Parallel processing for shuffled cross-correlations
        with Pool(num_cpus) as pool:
            shuffled_crosscorrs = pool.map(
                __compute_shuffled_crosscorr_both,
                [(gaussian_filter1d(m1_shuff, sigma), gaussian_filter1d(m2_shuff, sigma)) 
                 for m1_shuff, m2_shuff in zip(m1_shuffled_vectors, m2_shuffled_vectors)]
            )
        
        shuffled_crosscorrs_m1_m2 = np.array([s[0] for s in shuffled_crosscorrs])
        shuffled_crosscorrs_m2_m1 = np.array([s[1] for s in shuffled_crosscorrs])
        
        results.append({
            "session_name": session,
            "interaction_type": interaction,
            "run_number": run,
            "fixation_type": fixation_type,
            "crosscorr_m1_m2": crosscorr_m1_m2,
            "crosscorr_m2_m1": crosscorr_m2_m1,
            "mean_shuffled_m1_m2": np.mean(shuffled_crosscorrs_m1_m2, axis=0),
            "std_shuffled_m1_m2": np.std(shuffled_crosscorrs_m1_m2, axis=0),
            "mean_shuffled_m2_m1": np.mean(shuffled_crosscorrs_m2_m1, axis=0),
            "std_shuffled_m2_m1": np.std(shuffled_crosscorrs_m2_m1, axis=0)
        })
    
    return pd.DataFrame(results)

def __fft_crosscorrelation_both(x, y):
    """Compute cross-correlation using fftconvolve."""
    full_corr = fftconvolve(x, y[::-1], mode='full')

    n = len(x)
    mid = len(full_corr) // 2

    return full_corr[mid:mid + n], full_corr[:mid][::-1]

def __generate_shuffled_vectors(eye_mvm_behav_df, session, interaction, run, fixation_type, num_shuffles=20):
    """
    Generate shuffled versions of a binary fixation vector by randomly redistributing fixation intervals.
    """
    row = eye_mvm_behav_df[(eye_mvm_behav_df["session_name"] == session) &
                            (eye_mvm_behav_df["interaction_type"] == interaction) &
                            (eye_mvm_behav_df["run_number"] == run)].iloc[0]
    
    run_length = row["run_length"]
    categories = __categorize_fixations(row["fixation_location"])
    fixation_start_stop = row["fixation_start_stop"]
    
    # Get only the fixations of the requested type
    fixation_intervals = [
        (start, stop) for (start, stop), category in zip(fixation_start_stop, categories) if category == fixation_type
    ]
    fixation_durations = [stop - start + 1 for start, stop in fixation_intervals]
    total_fixation_duration = sum(fixation_durations)
    N = len(fixation_durations)
    
    # Compute total available non-fixation duration
    available_non_fixation_duration = run_length - total_fixation_duration
    
    shuffled_vectors = []
    for _ in range(num_shuffles):
        # Generate N+1 random non-fixation partitions that sum to available duration
        non_fixation_durations = np.random.multinomial(
            available_non_fixation_duration, 
            np.ones(N + 1) / (N + 1)
        )
        
        # Label fixation and non-fixation segments to ensure correct placement
        fixation_labels = [(dur, 1) for dur in fixation_durations]
        non_fixation_labels = [(dur, 0) for dur in non_fixation_durations]
        all_segments = fixation_labels + non_fixation_labels
        np.random.shuffle(all_segments)
        
        # Construct the shuffled binary vector
        shuffled_vector = np.zeros(run_length, dtype=int)
        index = 0
        for duration, label in all_segments:
            if index >= run_length:
                logging.warning("Index exceeded run length during shuffle generation")
                break
            if label == 1:
                shuffled_vector[index: index + duration] = 1  # Fixation interval
            index += duration
        
        shuffled_vectors.append(shuffled_vector)
    
    return shuffled_vectors

def __compute_shuffled_crosscorr_both(pair):
    """Helper function to compute cross-correlations for shuffled vectors."""
    return __fft_crosscorrelation_both(pair[0], pair[1])


# ** Call to main() **


if __name__ == "__main__":
    main()
