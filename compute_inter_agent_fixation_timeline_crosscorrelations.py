import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.stats import ttest_rel

import pdb

import load_data
import curate_data


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


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

    logger.info("Generating fix-related binary vectors")
    fix_binary_vector_df = generate_fixation_binary_vectors(eye_mvm_behav_df)
    logger.info("Fix-related binary vectors generated")
    behav_crosscorr_df = compute_crosscorr_and_shuffled_stats(
        fix_binary_vector_df, eye_mvm_behav_df, sigma=25, num_shuffles=20, num_cpus=8)




def generate_fixation_binary_vectors(df):
    """
    Generate a long-format dataframe with binary vectors for each fixation category 
    (eyes, face, out_of_roi), where each row corresponds to a run and fixation type.

    Args:
        df (pd.DataFrame): DataFrame with fixation start-stop indices and fixation locations.

    Returns:
        pd.DataFrame: A long-format DataFrame where each row represents a fixation type's binary vector.
    """
    binary_vectors = []

    for _, row in tqdm(df.iterrows(), desc="Processing df row"):
        run_length = row["run_length"]
        fixation_start_stop = row["fixation_start_stop"]
        fixation_categories = categorize_fixations(row["fixation_location"])

        # Initialize binary vectors
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

        # Append binary vectors in long format
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




def gaussian_smooth(vector, sigma=25):
    """Apply a Gaussian filter to smooth a binary fixation vector."""
    kernel_size = int(6 * sigma)  # 3-sigma rule
    kernel_size += 1 if kernel_size % 2 == 0 else 0  # Ensure odd-sized kernel
    kernel = gaussian(kernel_size, sigma)
    kernel /= kernel.sum()  # Normalize
    return convolve1d(vector, kernel, mode='constant')



def fft_crosscorrelation_both(x, y):
    """Compute cross-correlation using fftconvolve and extract positive lags."""
    full_corr = fftconvolve(x, y[::-1], mode='full')  # Full cross-correlation

    n = len(x)  # Original length
    mid = len(full_corr) // 2  # Midpoint (zero lag)

    crosscorr_m1_m2 = full_corr[mid:mid + n]  # Positive lags for m1 → m2
    crosscorr_m2_m1 = full_corr[:mid][::-1]   # Correctly extract and flip negative lags for m2 → m1

    return crosscorr_m1_m2, crosscorr_m2_m1



def generate_shuffled_vectors(original_vector, event_start_stop, num_shuffles=20):
    """Generate shuffled versions of a binary fixation vector by jittering event start times."""
    run_length = len(original_vector)
    shuffled_vectors = []
    max_shift = run_length - max([stop - start for start, stop in event_start_stop])  # Prevent out-of-bounds
    
    for _ in range(num_shuffles):
        shuffled_vector = np.zeros(run_length, dtype=int)
        for start, stop in event_start_stop:
            duration = stop - start
            new_start = np.random.randint(0, max_shift)
            new_stop = new_start + duration
            shuffled_vector[new_start:new_stop + 1] = 1
        shuffled_vectors.append(shuffled_vector)
    
    return shuffled_vectors


def compute_shuffled_crosscorr_both(pair):
    """Helper function to compute cross-correlations for shuffled vectors."""
    m1_shuff, m2_shuff, sigma = pair
    smoothed_m1 = gaussian_smooth(m1_shuff, sigma)
    smoothed_m2 = gaussian_smooth(m2_shuff, sigma)
    return fft_crosscorrelation_both(smoothed_m1, smoothed_m2)


def extract_start_stops_for_category(start_stops, categories, target_category):
    """
    Filter fixation start-stop pairs to only those matching the target category.
    
    Args:
        start_stops (list of lists): Original fixation start-stop pairs.
        categories (list): Categories assigned to each start-stop pair.
        target_category (str): The fixation type to filter for.

    Returns:
        list of lists: Filtered start-stop pairs matching the target category.
    """
    filtered_start_stops = [
        (start, stop)
        for (start, stop), category in zip(start_stops, categories)
        if (category == target_category) or (target_category == "face" and category in ["eyes", "non_eye_face"])
    ]
    return filtered_start_stops


def compute_crosscorr_and_shuffled_stats(df, eye_mvm_behav_df, sigma=25, num_shuffles=20, num_cpus=8):
    """
    Compute Gaussian-smoothed cross-correlation between m1 and m2's fixation behaviors,
    generate shuffled distributions in parallel, and store means & stds of shuffled cross-correlations.

    Args:
        df (pd.DataFrame): Dataframe containing binary fixation vectors.
        eye_mvm_behav_df (pd.DataFrame): Dataframe containing fixation start-stop indices and categories.
        sigma (int): Gaussian smoothing sigma.
        num_shuffles (int): Number of shuffled samples.
        num_cpus (int): Number of CPUs for parallel processing.

    Returns:
        pd.DataFrame: Dataframe with original cross-correlations and shuffled statistics.
    """
    results = []

    # Group by session, run, fixation type
    grouped = df.groupby(["session_name", "interaction_type", "run_number", "fixation_type"])

    for (session, interaction, run, fixation_type), group in grouped:
        if len(group) != 2:
            continue  # Skip if both m1 and m2 aren't present
        
        m1_row = group[group["agent"] == "m1"].iloc[0]
        m2_row = group[group["agent"] == "m2"].iloc[0]

        m1_vector = np.array(m1_row["binary_vector"])
        m2_vector = np.array(m2_row["binary_vector"])

        # Extract fixation start-stop times and categories from eye_mvm_behav_df
        m1_events = eye_mvm_behav_df[
            (eye_mvm_behav_df["session_name"] == session) &
            (eye_mvm_behav_df["interaction_type"] == interaction) &
            (eye_mvm_behav_df["run_number"] == run) &
            (eye_mvm_behav_df["agent"] == "m1")
        ].iloc[0]

        m2_events = eye_mvm_behav_df[
            (eye_mvm_behav_df["session_name"] == session) &
            (eye_mvm_behav_df["interaction_type"] == interaction) &
            (eye_mvm_behav_df["run_number"] == run) &
            (eye_mvm_behav_df["agent"] == "m2")
        ].iloc[0]

        # Filter start-stop pairs to the current fixation type
        m1_start_stops = extract_start_stops_for_category(
            m1_events["fixation_start_stop"],
            categorize_fixations(m1_events["fixation_location"]),
            fixation_type
        )
        m2_start_stops = extract_start_stops_for_category(
            m2_events["fixation_start_stop"],
            categorize_fixations(m2_events["fixation_location"]),
            fixation_type
        )

        # Apply Gaussian smoothing
        m1_smooth = gaussian_smooth(m1_vector, sigma)
        m2_smooth = gaussian_smooth(m2_vector, sigma)

        # Compute original cross-correlations using a single FFT
        crosscorr_m1_m2, crosscorr_m2_m1 = fft_crosscorrelation_both(m1_smooth, m2_smooth)
        pdb.set_trace()
        # Generate shuffled vectors using the filtered start-stop pairs
        shuffled_m1_vectors = generate_shuffled_vectors(m1_vector, m1_start_stops, num_shuffles)
        shuffled_m2_vectors = generate_shuffled_vectors(m2_vector, m2_start_stops, num_shuffles)

        # Parallel processing for shuffled cross-correlations
        with Pool(num_cpus) as pool:
            shuffled_crosscorrs = pool.map(
                compute_shuffled_crosscorr_both,
                [(m1_shuff, m2_shuff, sigma) for m1_shuff, m2_shuff in zip(shuffled_m1_vectors, shuffled_m2_vectors)]
            )
        pdb.set_trace()
        # Convert results to numpy arrays
        shuffled_crosscorrs_m1_m2 = np.array([s[0] for s in shuffled_crosscorrs])  # Extract m1 -> m2
        shuffled_crosscorrs_m2_m1 = np.array([s[1] for s in shuffled_crosscorrs])  # Extract m2 -> m1

        # Compute mean and std for each lag bin
        mean_shuffled_m1_m2 = np.mean(shuffled_crosscorrs_m1_m2, axis=0)
        std_shuffled_m1_m2 = np.std(shuffled_crosscorrs_m1_m2, axis=0)

        mean_shuffled_m2_m1 = np.mean(shuffled_crosscorrs_m2_m1, axis=0)
        std_shuffled_m2_m1 = np.std(shuffled_crosscorrs_m2_m1, axis=0)

        # Store results
        results.append({
            "session_name": session,
            "interaction_type": interaction,
            "run_number": run,
            "fixation_type": fixation_type,
            "crosscorr_m1_m2": crosscorr_m1_m2,
            "crosscorr_m2_m1": crosscorr_m2_m1,
            "mean_shuffled_m1_m2": mean_shuffled_m1_m2,
            "std_shuffled_m1_m2": std_shuffled_m1_m2,
            "mean_shuffled_m2_m1": mean_shuffled_m2_m1,
            "std_shuffled_m2_m1": std_shuffled_m2_m1
        })

    return pd.DataFrame(results)
