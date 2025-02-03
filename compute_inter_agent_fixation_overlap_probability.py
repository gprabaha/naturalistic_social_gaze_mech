import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.stats import ttest_rel, ranksums
import pingouin as pg

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
        'reanalyse_fixation_probabilities': True
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
        _analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params, group_by="monkey_pair")
        logger.info("Analysis and plotting complete")


# ** Sub-functions **


def _analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params, group_by="monkey_pair"):
    """Run the full pipeline of fixation probability analysis."""
    joint_probs_df = __compute_fixation_statistics(eye_mvm_behav_df, monkeys_per_session_df)
    __plot_joint_fixation_distributions(joint_probs_df, params, group_by=group_by)
    return joint_probs_df


def __compute_fixation_statistics(eye_mvm_behav_df, monkeys_per_session_df):
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
        
        m1_fixations = ___categorize_fixations(m1_df["fixation_location"].values[0])
        m2_fixations = ___categorize_fixations(m2_df["fixation_location"].values[0])
        run_length = m1_df["run_length"].values[0]
        
        for category in ["eyes", "non_eye_face", "face", "out_of_roi"]:
            if category != "face":
                m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fixations, m1_df["fixation_start_stop"].values[0]) if cat == category]
                m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fixations, m2_df["fixation_start_stop"].values[0]) if cat == category]
            else:
                m1_indices = [(start, stop) for cat, (start, stop) in zip(m1_fixations, m1_df["fixation_start_stop"].values[0]) if cat in {"eyes", "non_eye_face"}]
                m2_indices = [(start, stop) for cat, (start, stop) in zip(m2_fixations, m2_df["fixation_start_stop"].values[0]) if cat in {"eyes", "non_eye_face"}]
            
            joint_duration = ___compute_joint_duration(m1_indices, m2_indices)
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

def ___categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if set(fixes) & {"mouth", "face"} else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]

def ___compute_joint_duration(m1_indices, m2_indices):
    """Compute the overlapping duration between m1 and m2 fixation events."""
    m1_timepoints = set()
    for start, stop in m1_indices:
        m1_timepoints.update(range(start, stop + 1))

    m2_timepoints = set()
    for start, stop in m2_indices:
        m2_timepoints.update(range(start, stop + 1))
    
    joint_timepoints = m1_timepoints & m2_timepoints

    return len(joint_timepoints)



def __plot_joint_fixation_distributions(joint_prob_df, params, group_by="monkey_pair"):
    """Generate subplot comparisons for fixation probability distributions with multiple statistical tests."""
    logger.info("Generating fixation probability plots")
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_date += f"_{group_by}"
    root_dir = os.path.join(params['root_data_dir'], "plots", "inter_agent_fix_prob", today_date)
    os.makedirs(root_dir, exist_ok=True)

    for grouping_name, sub_df in tqdm(joint_prob_df.groupby(group_by), desc=f"Plotting {group_by}"):
        fig, axes = plt.subplots(3, 4, figsize=(16, 18))  # 3 rows, 4 columns layout
        axes = axes.reshape(3, 4)  # Ensure proper row-column alignment

        categories = ["eyes", "non_eye_face", "face", "out_of_roi"]

        # Loop through each category for plotting
        for i, category in enumerate(categories):
            cat_data = sub_df[sub_df["fixation_category"] == category]

            if not cat_data.empty:
                melted_data = cat_data.melt(id_vars=["fixation_category"],
                                            value_vars=["P(m1)*P(m2)", "P(m1&m2)"],
                                            var_name="Probability Type", value_name="Probability")

                # First row: Paired t-test
                t_stat_ttest, p_val_ttest = ttest_rel(cat_data["P(m1)*P(m2)"], cat_data["P(m1&m2)"])
                sns.violinplot(data=melted_data, x="Probability Type", y="Probability", ax=axes[0, i])
                axes[0, i].set_title(f"{category} Fixation Probabilities (Paired t-test)")
                axes[0, i].text(0.5, 0.9, f'p = {p_val_ttest:.4f}', ha='center', va='center', transform=axes[0, i].transAxes)

                # Second row: Mann-Whitney U test (Ranksum test)
                t_stat_ranksum, p_val_ranksum = ranksums(cat_data["P(m1)*P(m2)"], cat_data["P(m1&m2)"])
                sns.violinplot(data=melted_data, x="Probability Type", y="Probability", ax=axes[1, i])
                axes[1, i].set_title(f"{category} Fixation Probabilities (Ranksum test)")
                axes[1, i].text(0.5, 0.9, f'p = {p_val_ranksum:.4f}', ha='center', va='center', transform=axes[1, i].transAxes)

                # Third row: Bayesian t-test (Bayes Factor)
                bf = pg.bayesfactor_ttest(t_stat_ttest, cat_data.shape[0])
                # Determine color for Bayes Factor significance
                if bf < 3:
                    bf_color = "red"  # Weak evidence
                elif 3 <= bf < 10:
                    bf_color = "orange"  # Moderate evidence
                else:
                    bf_color = "green"  # Strong evidence
                sns.violinplot(data=melted_data, x="Probability Type", y="Probability", ax=axes[2, i])
                axes[2, i].set_title(f"{category} Fixation Probabilities (Bayesian t-test)")
                axes[2, i].text(0.5, 0.9, f'BF = {bf:.2f}', ha='center', va='center', transform=axes[2, i].transAxes, color=bf_color)
        
        plt.suptitle(f"{group_by.capitalize()}: {grouping_name} Fixation Probability Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, f"{grouping_name}_fixation_probabilities.png"))
        plt.close()


# ** Call to main() **

if __name__ == "__main__":
    main()
