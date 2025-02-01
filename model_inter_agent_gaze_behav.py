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

    if params.get('reanalyse_fixation_probabilities', False):
        logger.info("Analyzing fixation probabilities")
        analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params)
        logger.info("Analysis and plotting complete")
    
    # **Fixations as point processes**

    # analyze_fixations_as_point_processes(eye_mvm_behav_df)
    analyze_fixation_point_processes_pipeline(eye_mvm_behav_df, params)



    if params.get('reanalyze_fixation_crosscorrelations', False):
        logger.info("Generating fix-related binary vectors")
        fix_binary_vector_df = generate_fixation_binary_vectors(eye_mvm_behav_df)
        behav_crosscorr_df = compute_crosscorr_and_shuffled_stats(
            fix_binary_vector_df, eye_mvm_behav_df, sigma=25, num_shuffles=20, num_cpus=8)
        logger.info("Fix-related binary vectors generated")
    


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_inter_arrival_times(timepoints):
    """Compute inter-arrival times from sorted fixation midpoint times."""
    if len(timepoints) < 2:
        return []  # Not enough points for inter-arrival analysis
    timepoints = np.sort(timepoints)
    return np.diff(timepoints)

def fit_and_analyze_inter_arrival_times(inter_arrival_times):
    """Fit different distributions to inter-arrival times and return results."""
    if len(inter_arrival_times) < 2:
        return None  # Insufficient data

    # Fit exponential distribution
    exp_lambda = 1 / np.mean(inter_arrival_times)
    exp_fit = stats.expon.fit(inter_arrival_times)

    # Fit gamma distribution
    gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(inter_arrival_times)

    # Fit power-law (Pareto) distribution
    pareto_shape, pareto_loc, pareto_scale = stats.pareto.fit(inter_arrival_times, floc=0)

    # Compute Coefficient of Variation (CV)
    mean_ia = np.mean(inter_arrival_times)
    std_ia = np.std(inter_arrival_times)
    cv = std_ia / mean_ia if mean_ia > 0 else np.nan

    return {
        "exp_lambda": exp_lambda,
        "exp_fit_params": exp_fit,
        "gamma_params": (gamma_shape, gamma_loc, gamma_scale),
        "pareto_params": (pareto_shape, pareto_loc, pareto_scale),
        "mean": mean_ia,
        "std": std_ia,
        "cv": cv
    }


def plot_inter_arrival_distribution(inter_arrival_times, category, agent):
    """Plot histogram and fitted distributions for inter-arrival times."""
    if len(inter_arrival_times) < 2:
        print(f"Not enough data for {agent} - {category}. Skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    
    # Histogram of empirical data
    plt.hist(inter_arrival_times, bins=30, density=True, alpha=0.6, label="Empirical Data")

    # Generate x values for fitting
    x = np.linspace(min(inter_arrival_times), max(inter_arrival_times), 100)

    # Fit distributions
    exp_lambda = 1 / np.mean(inter_arrival_times)
    plt.plot(x, stats.expon.pdf(x, scale=1/exp_lambda), label="Exponential Fit", linestyle="--")

    gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(inter_arrival_times)
    plt.plot(x, stats.gamma.pdf(x, gamma_shape, loc=gamma_loc, scale=gamma_scale), label="Gamma Fit", linestyle="--")

    pareto_shape, pareto_loc, pareto_scale = stats.pareto.fit(inter_arrival_times, floc=0)
    plt.plot(x, stats.pareto.pdf(x, pareto_shape, loc=pareto_loc, scale=pareto_scale), label="Pareto Fit", linestyle="--")

    # Labels and legend
    plt.xlabel("Inter-Arrival Time")
    plt.ylabel("Density")
    plt.title(f"Inter-Arrival Time Distribution for {agent} - {category}")
    plt.legend()
    plt.show()


def analyze_fixations_as_point_processes(eye_mvm_behav_df):
    """Extract inter-arrival times for different fixation categories and analyze distributions."""
    grouped = list(eye_mvm_behav_df.groupby(["session_name", "interaction_type", "run_number"]))
    results = []

    for (session, interaction, run), sub_df in tqdm(grouped, desc="Processing sessions"):
        m1_df = sub_df[sub_df["agent"] == "m1"]
        m2_df = sub_df[sub_df["agent"] == "m2"]

        if m1_df.empty or m2_df.empty:
            continue

        m1_fixations = categorize_fixations(m1_df["fixation_location"].values[0])
        m2_fixations = categorize_fixations(m2_df["fixation_location"].values[0])

        for category in ["eyes", "non_eye_face", "out_of_roi"]:
            m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fixations, m1_df["fixation_start_stop"].values[0]) if cat == category]
            m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fixations, m2_df["fixation_start_stop"].values[0]) if cat == category]

            m1_times = [ (stop - start) / 2 for (start, stop) in m1_indices ]
            m2_times = [ (stop - start) / 2 for (start, stop) in m2_indices ]

            # Compute inter-arrival times
            m1_inter_arrivals = compute_inter_arrival_times(m1_times)
            m2_inter_arrivals = compute_inter_arrival_times(m2_times)

            # Fit and analyze distributions
            m1_analysis = fit_and_analyze_inter_arrival_times(m1_inter_arrivals)
            m2_analysis = fit_and_analyze_inter_arrival_times(m2_inter_arrivals)

            if m1_analysis:
                results.append({"session": session, "interaction": interaction, "run": run, "agent": "m1", "category": category, **m1_analysis})
                plot_inter_arrival_distribution(m1_inter_arrivals, category, "m1")

            if m2_analysis:
                results.append({"session": session, "interaction": interaction, "run": run, "agent": "m2", "category": category, **m2_analysis})
                plot_inter_arrival_distribution(m2_inter_arrivals, category, "m2")

    return results



def categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]







import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import scipy.stats as stats
from scipy.signal import correlate

def analyze_fixation_point_processes_pipeline(eye_mvm_behav_df, params):
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_date += "_by_run"
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_point_processes", today_date)
    os.makedirs(root_dir, exist_ok=True)

    grouped = list(eye_mvm_behav_df.groupby(["session_name", "interaction_type", "run_number"]))

    for (session, interaction, run), sub_df in tqdm(grouped, desc="Processing sessions"):
        session_dir = os.path.join(root_dir, f"{session}_{interaction}_run{run}")
        os.makedirs(session_dir, exist_ok=True)

        fixation_sequences = prepare_ordered_fixation_sequences(sub_df)
        inter_arrival_results = compute_labeled_inter_arrivals(fixation_sequences)

        for agent in ["m1", "m2"]:
            for category, inter_arrivals in inter_arrival_results[agent].items():
                if len(inter_arrivals) < 2:
                    continue

                plt.figure(figsize=(8, 5))
                sns.histplot(inter_arrivals, bins=30, kde=True, alpha=0.6, label="Empirical Data")
                exp_lambda = 1 / np.mean(inter_arrivals)
                x_vals = np.linspace(min(inter_arrivals), max(inter_arrivals), 100)
                plt.plot(x_vals, stats.expon.pdf(x_vals, scale=1/exp_lambda), label="Exponential Fit", linestyle="--")
                gamma_params = stats.gamma.fit(inter_arrivals)
                plt.plot(x_vals, stats.gamma.pdf(x_vals, *gamma_params), label="Gamma Fit", linestyle="--")
                plt.xlabel("Inter-Arrival Time")
                plt.ylabel("Density")
                plt.title(f"{session} - {interaction} - Run {run} | {agent.upper()} - {category}")
                plt.legend()
                save_path = os.path.join(session_dir, f"{agent}_{category}_inter_arrival.png")
                plt.savefig(save_path, dpi=300)
                plt.close()

        compute_cross_correlation(inter_arrival_results, session_dir, session, interaction, run)

    print(f"All plots saved in {root_dir}")










def prepare_ordered_fixation_sequences(sub_df):
    """
    Extract and arrange fixation event times by category for each agent in chronological order
    for a single session-run combination.
    
    Args:
        sub_df (DataFrame): Subset DataFrame for a specific session, interaction, and run.

    Returns:
        dict: A dictionary with fixation sequences for "m1" and "m2".
              Format:
              {
                  "m1": [(event_time, category), ...],
                  "m2": [(event_time, category), ...]
              }
    """
    fixation_sequences = {"m1": [], "m2": []}
    
    m1_df = sub_df[sub_df["agent"] == "m1"]
    m2_df = sub_df[sub_df["agent"] == "m2"]

    if m1_df.empty or m2_df.empty:
        return fixation_sequences  # Return empty if no data for either agent

    m1_fixations = categorize_fixations(m1_df["fixation_location"].values[0])
    m2_fixations = categorize_fixations(m2_df["fixation_location"].values[0])

    for category in ["eyes", "non_eye_face", "out_of_roi"]:
        m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fixations, m1_df["fixation_start_stop"].values[0]) if cat == category]
        m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fixations, m2_df["fixation_start_stop"].values[0]) if cat == category]

        for start, stop in m1_indices:
            event_time = (start + stop) / 2
            fixation_sequences["m1"].append((event_time, category))

        for start, stop in m2_indices:
            event_time = (start + stop) / 2
            fixation_sequences["m2"].append((event_time, category))

    # Sort fixations chronologically for each agent
    fixation_sequences["m1"].sort(key=lambda x: x[0])
    fixation_sequences["m2"].sort(key=lambda x: x[0])

    return fixation_sequences




def compute_labeled_inter_arrivals(fixation_sequences):
    """
    Compute inter-arrival times separately for each fixation category per agent.

    Args:
        fixation_sequences (dict): Dictionary with fixation events per agent.
    
    Returns:
        dict: Inter-arrival times for each agent and category.
    """
    inter_arrival_results = {"m1": {}, "m2": {}}

    for agent in ["m1", "m2"]:
        events = fixation_sequences[agent]
        category_times = {}

        for time, category in events:
            if category not in category_times:
                category_times[category] = []
            category_times[category].append(time)

        inter_arrival_results[agent] = {
            category: np.diff(np.sort(times)) for category, times in category_times.items() if len(times) > 1
        }

    return inter_arrival_results




import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

def analyze_inter_arrival_distributions(inter_arrival_results):
    """Fit distributions and compare inter-arrival times across categories."""
    
    for agent in ["m1", "m2"]:
        for category, inter_arrivals in inter_arrival_results[agent].items():
            if len(inter_arrivals) < 2:
                continue

            plt.figure(figsize=(8, 5))
            
            # Plot Histogram
            sns.histplot(inter_arrivals, bins=30, kde=True, alpha=0.6, label="Empirical Data")

            # Fit and plot Exponential
            exp_lambda = 1 / np.mean(inter_arrivals)
            x_vals = np.linspace(min(inter_arrivals), max(inter_arrivals), 100)
            plt.plot(x_vals, stats.expon.pdf(x_vals, scale=1/exp_lambda), label="Exponential Fit", linestyle="--")

            # Fit and plot Gamma
            gamma_params = stats.gamma.fit(inter_arrivals)
            plt.plot(x_vals, stats.gamma.pdf(x_vals, *gamma_params), label="Gamma Fit", linestyle="--")

            # Labels
            plt.xlabel("Inter-Arrival Time")
            plt.ylabel("Density")
            plt.title(f"{agent.upper()} - {category} Fixation Inter-Arrival")
            plt.legend()
            plt.show()



def compute_cross_correlation(inter_arrival_results, session_dir, session, interaction, run):
    """
    Compute and plot cross-correlations of inter-arrival times between m1 and m2 for each category.

    Args:
        inter_arrival_results (dict): Inter-arrival times for each agent and category.
        session_dir (str): Directory to save plots.
        session (str): Session name.
        interaction (str): Interaction type.
        run (int): Run number.
    """
    for category in ["eyes", "non_eye_face", "out_of_roi"]:
        if category in inter_arrival_results["m1"] and category in inter_arrival_results["m2"]:
            m1_ia = inter_arrival_results["m1"][category]
            m2_ia = inter_arrival_results["m2"][category]

            if len(m1_ia) > 1 and len(m2_ia) > 1:
                min_length = min(len(m1_ia), len(m2_ia))
                m1_ia, m2_ia = m1_ia[:min_length], m2_ia[:min_length]

                cross_corr = np.correlate(m1_ia - np.mean(m1_ia), m2_ia - np.mean(m2_ia), mode='full')
                cross_corr /= np.max(cross_corr)  # Normalize

                # Create cross-correlation plot
                plt.figure(figsize=(8, 5))
                lags = np.arange(-min_length + 1, min_length)
                plt.plot(lags, cross_corr, label=f"Cross-Correlation ({category})")
                plt.axvline(0, color='black', linestyle="--", alpha=0.7)
                plt.xlabel("Lag")
                plt.ylabel("Normalized Correlation")
                plt.title(f"Cross-Correlation: {session} - {interaction} - Run {run} - {category}")
                plt.legend()

                save_path = os.path.join(session_dir, f"cross_correlation_{category}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()








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


# Example usage
# crosscorr_results_df = compute_crosscorr_and_shuffled_stats(fix_binary_vector_df, sigma=50, num_shuffles=100, num_cpus=8)



# ** MAIN **

if __name__ == "__main__":
    main()
