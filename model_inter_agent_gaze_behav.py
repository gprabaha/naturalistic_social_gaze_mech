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
        'reanalyze_fixation_crosscorrelations': True
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

    if params.get('reanalyse_fixation_probabilities', False):
        logger.info("Analyzing fixation probabilities")
        analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params)
        logger.info("Analysis and plotting complete")
    
    if params.get('reanalyze_fixation_crosscorrelations', False):
        logger.info("Generating fix-related binary vectors")
        fix_binary_vector_df = generate_fixation_binary_vectors(eye_mvm_behav_df)
        pdb.set_trace()
        logger.info("Fix-related binary vectors generated")

    


def analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params):
    """Run the full pipeline of fixation probability analysis."""
    joint_probs_df = compute_fixation_statistics(eye_mvm_behav_df, monkeys_per_session_df)
    plot_joint_fixation_distributions(joint_probs_df, params)
    return joint_probs_df


def compute_fixation_statistics(eye_mvm_behav_df, monkeys_per_session_df):
    """Compute fixation probabilities and joint fixation probabilities for m1 and m2."""
    logger.info("Computing fixation statistics")
    joint_probs = []
    
    grouped = list(eye_mvm_behav_df.groupby(["session_name", "interaction_type", "run_number"]))
    for (session, interaction, run), sub_df in tqdm(grouped, desc="Processing sessions"):
        m1 = monkeys_per_session_df[monkeys_per_session_df["session_name"] == session]["m1"].iloc[0]
        m2 = monkeys_per_session_df[monkeys_per_session_df["session_name"] == session]["m2"].iloc[0]
        m1_df = sub_df[sub_df["agent"] == "m1"]
        m2_df = sub_df[sub_df["agent"] == "m2"]
        
        if m1_df.empty or m2_df.empty:
            continue
        
        m1_fixations = categorize_fixations(m1_df["fixation_location"].values[0])
        m2_fixations = categorize_fixations(m2_df["fixation_location"].values[0])
        run_length = m1_df["run_length"].values[0]
        
        for category in ["eyes", "non_eye_face", "face", "out_of_roi"]:
            if category != "face":
                m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fixations, m1_df["fixation_start_stop"].values[0]) if cat == category]
                m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fixations, m2_df["fixation_start_stop"].values[0]) if cat == category]
            else:
                m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fixations, m1_df["fixation_start_stop"].values[0]) if cat in {"eyes", "non_eye_face"}]
                m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fixations, m2_df["fixation_start_stop"].values[0]) if cat in {"eyes", "non_eye_face"}]
            
            joint_duration = compute_joint_duration(m1_indices, m2_indices)
            p_m1 = sum(stop + 1 - start for start, stop in m1_indices) / run_length
            p_m2 = sum(stop + 1 - start for start, stop in m2_indices) / run_length
            p_joint = joint_duration / run_length
            
            joint_probs.append({
                "monkey_pair": f"{m1}-{m2}",
                "session_name": session, "interaction_type": interaction, "run_number": run,
                "fixation_category": category, "P(m1)": p_m1, "P(m2)": p_m2,
                "P(m1)*P(m2)": p_m1 * p_m2, "P(m1&m2)": p_joint
            })

    logger.info("Fixation statistics computation complete")
    return pd.DataFrame(joint_probs)


def compute_joint_duration(m1_indices, m2_indices):
    """Compute the overlapping duration between m1 and m2 fixation events."""
    m1_timepoints = set()
    for start, stop in m1_indices:
        m1_timepoints.update(range(start, stop + 1))

    m2_timepoints = set()
    for start, stop in m2_indices:
        m2_timepoints.update(range(start, stop + 1))
    
    joint_timepoints = m1_timepoints & m2_timepoints

    return len(joint_timepoints)



def categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]


def plot_joint_fixation_distributions(joint_prob_df, params):
    """Generate subplot comparisons for fixation probability distributions."""
    logger.info("Generating fixation probability plots")
    today_date = datetime.today().strftime('%Y-%m-%d')
    group_by = "monkey_pair"
    today_date += f"_{group_by}"
    root_dir = os.path.join(params['root_data_dir'], "plots", "inter_agent_fix_prob", today_date)
    os.makedirs(root_dir, exist_ok=True)

    for grouping_name, sub_df in tqdm(joint_prob_df.groupby(group_by), desc=f"Plotting {group_by}"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, category in enumerate(["eyes", "non_eye_face", "face", "out_of_roi"]):
            cat_data = sub_df[sub_df["fixation_category"] == category]
            
            if not cat_data.empty:
                sns.violinplot(data=cat_data.melt(id_vars=["fixation_category"],
                                                  value_vars=["P(m1)*P(m2)", "P(m1&m2)"],
                                                  var_name="Probability Type", value_name="Probability"),
                               x="Probability Type", y="Probability", ax=axes[i])
                axes[i].set_title(f"{category} Fixation Probabilities")
                
                if "P(m1)*P(m2)" in cat_data and "P(m1&m2)" in cat_data:
                    t_stat, p_val = ttest_rel(cat_data["P(m1)*P(m2)"], cat_data["P(m1&m2)"])
                    axes[i].text(0.5, 0.9, f'p = {p_val:.4f}', ha='center', va='center', transform=axes[i].transAxes)
        
        plt.suptitle(f"{group_by.capitalize()}: {grouping_name} Fixation Probability Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, f"{grouping_name}_fixation_probabilities.png"))
        plt.close()
    
    logger.info("Plot generation complete")


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




import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, gaussian
from scipy.ndimage import convolve1d
from multiprocessing import Pool


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

        # Generate shuffled vectors using the filtered start-stop pairs
        shuffled_m1_vectors = generate_shuffled_vectors(m1_vector, m1_start_stops, num_shuffles)
        shuffled_m2_vectors = generate_shuffled_vectors(m2_vector, m2_start_stops, num_shuffles)

        # Parallel processing for shuffled cross-correlations
        with Pool(num_cpus) as pool:
            shuffled_crosscorrs = pool.map(
                compute_shuffled_crosscorr_both,
                [(m1_shuff, m2_shuff, sigma) for m1_shuff, m2_shuff in zip(shuffled_m1_vectors, shuffled_m2_vectors)]
            )

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


# Example usage
# crosscorr_results_df = compute_crosscorr_and_shuffled_stats(fix_binary_vector_df, sigma=50, num_shuffles=100, num_cpus=8)



# ** MAIN **

if __name__ == "__main__":
    main()
