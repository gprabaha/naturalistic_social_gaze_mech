import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.stats import ttest_rel, wilcoxon, shapiro
import pingouin as pg
from IPython.display import display

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
        'reanalyse_fixation_probabilities': True,
        'use_time_chunks': True
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
        analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params, group_by="session_name")
        logger.info("Analysis and plotting complete")


# ** Sub-functions **


def analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params, group_by="monkey_pair"):
    """Run the full pipeline of fixation probability analysis."""
    joint_probs_df = compute_fixation_statistics(eye_mvm_behav_df, monkeys_per_session_df, params)
    plot_joint_fixation_distributions(joint_probs_df, params, group_by=group_by)
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


def categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]

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


def plot_joint_fixation_distributions(joint_prob_df, params, group_by="monkey_pair"):
    """Generate subplot comparisons for fixation probability distributions in a Jupyter Notebook."""
    logger.info("Generating fixation probability plots")
    
    for grouping_name, sub_df in tqdm(joint_prob_df.groupby(group_by), desc=f"Plotting {group_by}"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))  # 1 row, 4 columns layout
        categories = ["eyes", "non_eye_face", "face", "out_of_roi"]

        for i, category in enumerate(categories):
            cat_data = sub_df[sub_df["fixation_category"] == category]

            if not cat_data.empty:
                melted_data = cat_data.melt(id_vars=["fixation_category"],
                                            value_vars=["P(m1)*P(m2)", "P(m1&m2)"],
                                            var_name="Probability Type", value_name="Probability")

                shapiro_tests_m1_times_m2 = shapiro(cat_data["P(m1)*P(m2)"])[1]
                shapiro_tests_m1_and_m2 = shapiro(cat_data["P(m1&m2)"])[1]
                is_normal = (shapiro_tests_m1_times_m2 > 0.05) & (shapiro_tests_m1_and_m2 > 0.05)

                if is_normal:
                    t_stat, p_val_ttest = ttest_rel(cat_data["P(m1)*P(m2)"], cat_data["P(m1&m2)"])
                    bf = pg.bayesfactor_ttest(t_stat, cat_data.shape[0])
                    test_result_text = f"T-test p = {p_val_ttest:.4f}\nBF = {bf:.2f}"
                else:
                    _, p_val_wilcoxon = wilcoxon(cat_data["P(m1)*P(m2)"], cat_data["P(m1&m2)"])
                    test_result_text = f"Wilcoxon p = {p_val_wilcoxon:.4f}"

                sns.violinplot(data=melted_data, x="Probability Type", y="Probability", ax=axes[i], hue="Probability Type")
                axes[i].set_title(f"{category} Fixation Probabilities")
                axes[i].text(0.5, 0.9, test_result_text, ha='center', va='center', 
                             transform=axes[i].transAxes, fontsize=10)

        plt.suptitle(f"{group_by.capitalize()}: {grouping_name} Fixation Probability Distributions")
        plt.tight_layout()
        plt.show()  # Display plot inline in Jupyter Notebook
        # display(fig)



# ** Call to main() **

if __name__ == "__main__":
    main()
