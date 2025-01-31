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
    params = {'xx': 0}
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
    logger.info("Loading data files")
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    logger.info("Data loaded successfully")
    
    logger.info("Analyzing fixation probabilities")
    analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, params)
    logger.info("Analysis and plotting complete")
    
    pdb.set_trace()
    return 0


def analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, params):
    """Run the full pipeline of fixation probability analysis."""
    joint_probs_df = compute_fixation_statistics(eye_mvm_behav_df)
    plot_joint_fixation_distributions(joint_probs_df, params)
    return joint_probs_df


def compute_fixation_statistics(df):
    """Compute fixation probabilities and joint fixation probabilities for m1 and m2."""
    logger.info("Computing fixation statistics")
    joint_probs = []
    
    grouped = list(df.groupby(["session_name", "interaction_type", "run_number"]))
    for (session, interaction, run), sub_df in tqdm(grouped, desc="Processing sessions"):
        m1_df = sub_df[sub_df["agent"] == "m1"]
        m2_df = sub_df[sub_df["agent"] == "m2"]
        
        if m1_df.empty or m2_df.empty:
            continue
        
        m1_fixations = categorize_fixations(m1_df["fixation_location"].values[0])
        m2_fixations = categorize_fixations(m2_df["fixation_location"].values[0])
        run_length = m1_df["run_length"].values[0]
        
        for category in ["eyes", "non_eye_face", "face"]:
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
                "session_name": session, "interaction_type": interaction, "run_number": run,
                "fixation_category": category, "P(m1)": p_m1, "P(m2)": p_m2,
                "P(m1)*P(m2)": p_m1 * p_m2, "P(m1&m2)": p_joint
            })
    
    logger.info("Fixation statistics computation complete")
    return pd.DataFrame(joint_probs)


def compute_joint_duration(m1_indices, m2_indices):
    """Compute the overlapping duration between m1 and m2 fixation events."""
    
    # Generate sets of all time indices for m1 fixations
    m1_timepoints = set()
    for start, stop in m1_indices:
        m1_timepoints.update(range(start, stop + 1))

    # Generate sets of all time indices for m2 fixations
    m2_timepoints = set()
    for start, stop in m2_indices:
        m2_timepoints.update(range(start, stop + 1))
    
    # Compute the intersection of both sets
    joint_timepoints = m1_timepoints & m2_timepoints

    # Return the size of the intersection
    return len(joint_timepoints)



def categorize_fixations(fix_locations):
    """Categorize fixation locations into predefined categories."""
    return [
        "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
        "non_eye_face" if "face" in set(fixes) else
        "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
        for fixes in fix_locations
    ]


def plot_joint_fixation_distributions(joint_prob_df, params):
    """Generate subplot comparisons for fixation probability distributions."""
    logger.info("Generating fixation probability plots")
    today_date = datetime.today().strftime('%Y-%m-%d')
    root_dir = os.path.join(params['root_data_dir'], "plots", "inter_agent_fix_prob", today_date)
    os.makedirs(root_dir, exist_ok=True)
    
    for session, sub_df in tqdm(joint_prob_df.groupby("session_name"), desc="Plotting sessions"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, category in enumerate(["eyes", "non_eye_face", "face"]):
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
        
        plt.suptitle(f"Session {session} Fixation Probability Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, f"{session}_fixation_probabilities.png"))
        plt.close()
    
    logger.info("Plot generation complete")


# ** MAIN **

if __name__ == "__main__":
    main()
