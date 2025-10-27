import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import scipy.stats as stats
from scipy.signal import correlate

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

    analyze_fixation_point_processes_pipeline(eye_mvm_behav_df, params)



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


def categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]


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




if __name__ == "__main__":
    main()



## ** ARCHIVE **

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