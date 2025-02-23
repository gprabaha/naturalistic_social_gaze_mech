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
        'is_cluster': False,
        'reanalyse_fixation_probabilities': False,
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

    logger.info("Plotting best runs timeline")
    best_face_run, best_out_run = combined_timeline_plots_for_optimal_runs(eye_mvm_behav_df, monkeys_per_session_df, params)
    logger.info("Plotting best runs timeline finished")
    
    pdb.set_trace()

    return 0

# ** Sub-functions **


def analyze_and_plot_fixation_probabilities(eye_mvm_behav_df, monkeys_per_session_df, params, group_by="monkey_pair"):
    """Run the full pipeline of fixation probability analysis."""
    joint_probs_df = compute_fixation_statistics(eye_mvm_behav_df, monkeys_per_session_df, params)
    plot_joint_fixation_distributions(joint_probs_df, params, group_by=group_by)

    # plot_combined_fixation_distributions(joint_probs_df, params)
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


def combined_timeline_plots_for_optimal_runs(eye_mvm_behav_df, monkeys_per_session_df, params):
    """
    Computes fixation joint probabilities and selects:
      - the run with the highest difference (P(m1&m2) - P(m1)*P(m2)) for face fixations,
      - the run with the difference closest to zero for out_of_roi fixations.
      
    Then creates a single schematic graphic with two columns (face fixations on the left, 
    out_of_roi fixations on the right) and four rows (M1 timeline, M2 timeline, overlay, 
    and overlap) that displays only the timeline bars and textual identifiers without any axes.
    
    Different color sets are used for each fixation category.
    
    The schematic is saved as a PDF with a transparent background in:
      params['root_data_dir']/plots/best_fix_prob_timeline/date_dir/fig_name.pdf
    """
    # Compute joint probabilities.
    joint_df = compute_fixation_statistics(eye_mvm_behav_df, monkeys_per_session_df)
    joint_df["diff"] = joint_df["P(m1&m2)"] - joint_df["P(m1)*P(m2)"]

    # Select runs.
    face_df = joint_df[joint_df["fixation_category"] == "face"]
    best_face_run = face_df.loc[face_df["diff"].idxmax()]

    out_df = joint_df[joint_df["fixation_category"] == "out_of_roi"].copy()
    out_df["abs_diff"] = out_df["diff"].abs()
    best_out_run = out_df.loc[out_df["abs_diff"].idxmin()]

    # Extract fixation intervals.
    face_fix_data = _extract_run_fixations(eye_mvm_behav_df, best_face_run, fixation_category="face")
    out_fix_data  = _extract_run_fixations(eye_mvm_behav_df, best_out_run, fixation_category="out_of_roi")

    # Define two distinct color schemes.
    # Face fixation colors.
    face_m1_color = "#0072B2"      # blue
    face_m2_color = "#E69F00"      # orange
    face_overlap_color = "#009E73" # green

    # Out-of-ROI fixation colors.
    out_m1_color = "#D55E00"       # reddish
    out_m2_color = "#CC79A7"       # purple
    out_overlap_color = "#F0E442"  # yellow

    # Create a figure with 4 rows x 2 columns.
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 12))
    
    # Row 1: M1 timeline.
    _plot_intervals(axes[0,0], face_fix_data["m1"], face_fix_data["run_length"], color=face_m1_color)
    _plot_intervals(axes[0,1], out_fix_data["m1"], out_fix_data["run_length"], color=out_m1_color)
    
    # Row 2: M2 timeline.
    _plot_intervals(axes[1,0], face_fix_data["m2"], face_fix_data["run_length"], color=face_m2_color)
    _plot_intervals(axes[1,1], out_fix_data["m2"], out_fix_data["run_length"], color=out_m2_color)
    
    # Row 3: Overlay timeline (M1 and M2).
    _plot_intervals(axes[2,0], face_fix_data["m1"], face_fix_data["run_length"], color=face_m1_color, y_pos=10)
    _plot_intervals(axes[2,0], face_fix_data["m2"], face_fix_data["run_length"], color=face_m2_color, y_pos=5)
    _plot_intervals(axes[2,1], out_fix_data["m1"], out_fix_data["run_length"], color=out_m1_color, y_pos=10)
    _plot_intervals(axes[2,1], out_fix_data["m2"], out_fix_data["run_length"], color=out_m2_color, y_pos=5)
    
    # Row 4: Overlap only.
    _plot_intervals(axes[3,0], face_fix_data["overlap"], face_fix_data["run_length"], color=face_overlap_color)
    _plot_intervals(axes[3,1], out_fix_data["overlap"], out_fix_data["run_length"], color=out_overlap_color)

    # Remove all axes lines, ticks, and labels.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("none")

    # Add global column labels at the top.
    fig.text(0.25, 0.95, "Face Fixations\n{} Run {}".format(best_face_run["session_name"],
                                                               best_face_run["run_number"]),
             ha="center", fontsize=14, fontweight='bold')
    fig.text(0.75, 0.95, "Out-of-ROI Fixations\n{} Run {}".format(best_out_run["session_name"],
                                                                  best_out_run["run_number"]),
             ha="center", fontsize=14, fontweight='bold')
    
    # Add row labels on the left margin.
    row_labels = ["M1 Timeline", "M2 Timeline", "Overlay", "Overlap"]
    # Approximate vertical positions (figure fraction).
    row_positions = [0.82, 0.62, 0.42, 0.22]
    for pos, label in zip(row_positions, row_labels):
        fig.text(0.05, pos, label, va="center", ha="left", fontsize=14, fontweight="bold")
    
    # Save the schematic as a PDF with a transparent background.
    _finalize_and_save(fig, params, "combined_fixation_schematic")
    plt.close(fig)
    
    return best_face_run, best_out_run

def _extract_run_fixations(eye_mvm_behav_df, run_info, fixation_category):
    """
    For a given run (run_info) and fixation_category, extract fixation intervals for m1 and m2.
    Returns a dictionary with:
       - run_length,
       - m1: list of intervals,
       - m2: list of intervals,
       - overlap: list of overlapping intervals between m1 and m2.
    """
    session = run_info["session_name"]
    interaction = run_info["interaction_type"]
    run_number = run_info["run_number"]
    
    run_data = eye_mvm_behav_df[
        (eye_mvm_behav_df["session_name"] == session) &
        (eye_mvm_behav_df["interaction_type"] == interaction) &
        (eye_mvm_behav_df["run_number"] == run_number)
    ]
    
    fix_data = {}
    run_length = None
    for agent in ["m1", "m2"]:
        agent_row = run_data[run_data["agent"] == agent]
        if agent_row.empty:
            fix_data[agent] = []
            continue
        agent_row = agent_row.iloc[0]
        run_length = agent_row["run_length"]
        fixation_locations = agent_row["fixation_location"]
        fixation_intervals = agent_row["fixation_start_stop"]
        categories = categorize_fixations(fixation_locations)
        if fixation_category == "face":
            selected = [(start, stop) for cat, (start, stop) in zip(categories, fixation_intervals)
                        if cat in {"eyes", "non_eye_face"}]
        else:
            selected = [(start, stop) for cat, (start, stop) in zip(categories, fixation_intervals)
                        if cat == fixation_category]
        fix_data[agent] = selected

    overlap_intervals = _compute_overlap(fix_data.get("m1", []), fix_data.get("m2", []))
    
    return {"run_length": run_length, "m1": fix_data.get("m1", []), "m2": fix_data.get("m2", []), "overlap": overlap_intervals}

def _plot_intervals(ax, intervals, run_length, color, y_pos=0, height=5):
    """
    Plot fixation intervals as horizontal bars using broken_barh.
    Each interval is a tuple (start, stop) and the width is computed as stop - start + 1.
    y_pos can be adjusted to place bars on different vertical positions within the same axis.
    """
    bars = [(start, stop - start + 1) for start, stop in intervals]
    if bars:
        ax.broken_barh(bars, (y_pos, height), facecolors=color)
    else:
        ax.text(run_length/2, y_pos + height/2, "No fixations", ha="center", va="center", color="gray")

def _compute_overlap(intervals1, intervals2):
    """
    Compute intersection intervals between two lists of intervals.
    Each interval is a tuple (start, stop). Returns a list of (start, stop) pairs.
    """
    intersections = []
    for s1, e1 in intervals1:
        for s2, e2 in intervals2:
            start = max(s1, s2)
            end = min(e1, e2)
            if start <= end:
                intersections.append((start, end))
    return intersections

def _finalize_and_save(fig, params, fig_name):
    """
    Finalize the figure settings and save as a PDF with a transparent background.
    The file is saved in:
       params['root_data_dir']/plots/best_fix_prob_timeline/date_dir/fig_name.pdf
    """
    fig.patch.set_alpha(0)
    for ax in fig.get_axes():
        ax.set_facecolor("none")
    date_dir = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join(params['root_data_dir'], "plots", "best_fix_prob_timeline", date_dir)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, fig_name + ".pdf")
    fig.savefig(file_path, format="pdf", transparent=True, bbox_inches="tight")


    


# ** Call to main() **

if __name__ == "__main__":
    main()
