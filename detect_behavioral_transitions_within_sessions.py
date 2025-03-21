import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from scipy.stats import f_oneway
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests

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
        'neural_data_bin_size': 0.01,  # 10 ms in seconds
        'smooth_spike_counts': True,
        'gaussian_smoothing_sigma': 2,
        'time_window_before_and_after_event_for_psth': 0.5,
        'min_consecutive_sig_bins': 5,
        'min_total_sig_bins': 25
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    logger.info("Parameters initialized successfully")
    return params


def main():
    logger.info("Starting the script")
    params = _initialize_params()
    sparse_nan_removed_sync_gaze_data_df_filepath = os.path.join(
        params['processed_data_dir'], 'sparse_nan_removed_sync_gaze_data_df.pkl'
    )
    eye_mvm_behav_df_file_path = os.path.join(
        params['processed_data_dir'], 'eye_mvm_behav_df.pkl'
    )
    spike_times_file_path = os.path.join(
        params['processed_data_dir'], 'spike_times_df.pkl'
    )
    logger.info("Loading data files")
    sparse_nan_removed_sync_gaze_df = load_data.get_data_df(sparse_nan_removed_sync_gaze_data_df_filepath)
    eye_mvm_behav_df = load_data.get_data_df(eye_mvm_behav_df_file_path)
    spike_times_df = load_data.get_data_df(spike_times_file_path)
    _compute_spiking_for_various_transitions(eye_mvm_behav_df, spike_times_df, sparse_nan_removed_sync_gaze_df, params)



def _compute_spiking_for_various_transitions(eye_mvm_behav_df, spike_times_df, sparse_nan_removed_sync_gaze_df, params):
    logger.info("Computing transition probabilities and spiking responses")
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_date += "_5-25_minbin"
    root_dir = os.path.join(params['root_data_dir'], "plots", "fixation_transition_spiking", today_date)
    os.makedirs(root_dir, exist_ok=True)
    all_statistical_summaries = []
    for session_name, session_behav_df in tqdm(eye_mvm_behav_df.groupby("session_name"), desc="Processing Sessions"):
        session_spike_df = spike_times_df[spike_times_df["session_name"] == session_name]
        session_gaze_df = sparse_nan_removed_sync_gaze_df[sparse_nan_removed_sync_gaze_df["session_name"] == session_name]
        session_summary = __process_session(session_name, session_behav_df, session_spike_df, session_gaze_df, root_dir, params)
        all_statistical_summaries.extend(session_summary)
    # Plot region summary across all sessions
    __plot_region_summary(all_statistical_summaries, spike_times_df, root_dir)


def __process_session(session_name, session_behav_df, session_spike_df, session_gaze_df, root_dir, params):
    logger.info(f"Processing session {session_name}")
    statistical_summary = []
    for _, unit in tqdm(session_spike_df.iterrows(), total=len(session_spike_df), desc=f"Processing units in {session_name}"):
        # if unit["unit_uuid"] == 
        unit_uuid, brain_region, significant_results = __plot_fixation_transition_spiking(
            unit, session_behav_df, session_gaze_df, root_dir, params
        )
        if significant_results:
            statistical_summary.append((unit_uuid, brain_region, significant_results))
    return statistical_summary


def __plot_fixation_transition_spiking(unit, session_behav_df, session_gaze_df, root_dir, params):
    """Plots fixation transition spiking and runs statistical tests for both previous and next fixation transitions."""
    unit_uuid = unit["unit_uuid"]
    brain_region = unit["region"]
    spike_times = np.array(unit["spike_ts"])
    rois = ["eyes", "non_eye_face", "object", "out_of_roi"]
    transitions = ["from", "to"]
    min_consecutive_sig_bins = params.get("min_consecutive_sig_bins", 9)  # Default to 5 if not specified
    min_total_sig_bins = params.get("min_total_sig_bins", 90)

    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # First two rows: Mean spiking
    axs = [[fig.add_subplot(gs[row, col]) for col in range(4)] for row in range(2)]  
    # Third row: ROI-Level Significance Matrix
    significance_ax = fig.add_subplot(gs[2, :])  

    statistical_results = {}

    for row_idx, transition_type in enumerate(transitions):  # 'from' is row 0, 'to' is row 1
        for col_idx, roi in enumerate(rois):  # ROIs are ordered in columns
            spike_data = []  # Store (spike_counts, transition_label) tuples
            timeline = None

            for trans in rois:
                trial_spike_counts, timeline = ___compute_spiking_per_trial(
                    roi, trans, session_behav_df, session_gaze_df, spike_times, params, transition_type
                )
                if trial_spike_counts:
                    spike_data.extend(trial_spike_counts)

            if timeline is not None:
                significant_bins = ___perform_anova_and_mark_plot(spike_data, timeline, axs[row_idx][col_idx])

                # Ensure we check for **at least min_consecutive_sig_bins in a row**
                if len(significant_bins) > 0:
                    # **Fixed Consecutive Check: Identify longest run of consecutive bins**
                    groups = np.split(significant_bins, np.where(np.diff(significant_bins) > 1)[0] + 1)
                    longest_run = max(len(g) for g in groups)  # Ensure last run is considered
                    if (longest_run >= min_consecutive_sig_bins) or (len(significant_bins) >= min_total_sig_bins):
                        statistical_results.setdefault(roi, {}).setdefault(transition_type, []).append(unit_uuid)

            if spike_data:
                spike_df = pd.DataFrame(spike_data, columns=["spike_counts", "transition"])
                mean_spiking = {
                    transition: np.mean(np.array(data.tolist()), axis=0)
                    for transition, data in spike_df.groupby("transition")["spike_counts"]
                }
                for transition, mean_values in mean_spiking.items():
                    axs[row_idx][col_idx].plot(timeline[:-1], mean_values, label=transition)

            axs[row_idx][col_idx].set_title(f"{transition_type.capitalize()} Transition: {roi}")
            axs[row_idx][col_idx].legend()

    # Create significance matrix (Rows = transitions, Columns = ROIs, matching subplot layout)
    significance_matrix = np.zeros((len(transitions), len(rois)))  # (From/To x ROIs)

    for row_idx, transition_type in enumerate(transitions):
        for col_idx, roi in enumerate(rois):
            if roi in statistical_results and transition_type in statistical_results[roi]:
                significance_matrix[row_idx, col_idx] = 1  # Rows match transitions, columns match ROIs

    # **Fixed: Ensure white for non-significant, black for significant**
    im = significance_ax.imshow(significance_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)  
    significance_ax.set_xticks(range(len(rois)))
    significance_ax.set_xticklabels(rois)
    significance_ax.set_yticks(range(len(transitions)))
    significance_ax.set_yticklabels(transitions)
    significance_ax.set_title("ROI-Level Significance Matrix")

    plt.suptitle(f"Fixation Transition Spiking for UUID {unit_uuid}")
    plt.tight_layout()

    # Save plots
    save_dir = os.path.join(root_dir, "sig_units") if statistical_results else root_dir
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"fixation_transitions_{unit_uuid}.png"), dpi=100)
    plt.close(fig)

    return unit_uuid, brain_region, statistical_results



def ___compute_spiking_per_trial(roi, transition, session_behav_df, session_gaze_df, spike_times, params, transition_type):
    """Computes spike counts per trial for a given transition type."""
    spike_counts_per_trial = []
    bin_size = params["neural_data_bin_size"]
    sigma = params['gaussian_smoothing_sigma']
    time_window = params['time_window_before_and_after_event_for_psth']
    timeline = np.arange(-time_window, time_window + bin_size, bin_size)
    m1_behav_df = session_behav_df[session_behav_df["agent"] == 'm1']
    for _, run_row in m1_behav_df.iterrows():
        fixations = run_row["fixation_location"]
        categorized_fixations = [
            "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
            "non_eye_face" if "face" in set(fixes) else
            "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
            for fixes in fixations
        ]
        run_number = run_row["run_number"]
        if transition_type == "to":
            relevant_fixations = [
                i for i, fix_label in enumerate(categorized_fixations[:-1])
                if fix_label == roi and categorized_fixations[i + 1] == transition
            ]
        else:
            relevant_fixations = [
                i for i, fix_label in enumerate(categorized_fixations[1:])
                if fix_label == roi and categorized_fixations[i - 1] == transition
            ]
        if not relevant_fixations:
            continue
        run_gaze_data = session_gaze_df[
            (session_gaze_df["run_number"] == run_number) & (session_gaze_df["agent"] == 'm1')
        ]
        if run_gaze_data.empty:
            continue
        neural_timeline = run_gaze_data.iloc[0]["neural_timeline"]
        for fixation_idx in relevant_fixations:
            fixation_start = run_row["fixation_start_stop"][fixation_idx][0]
            if fixation_start >= len(neural_timeline):
                continue
            fixation_time = neural_timeline[fixation_start]
            bins = np.linspace(fixation_time - time_window, fixation_time + time_window, int(2 * time_window / bin_size) + 1).ravel()
            spike_counts, _ = np.histogram(spike_times, bins=bins)
            spike_counts = spike_counts / bin_size
            if params["smooth_spike_counts"]:
                spike_counts = gaussian_filter1d(spike_counts, sigma=sigma)
            if transition_type == "to":
                spike_counts_per_trial.append((spike_counts, f"{roi} → {transition}"))
            else:
                spike_counts_per_trial.append((spike_counts, f"{transition} → {roi}"))
    return spike_counts_per_trial, timeline


def ___perform_anova_and_mark_plot(spike_data, timeline, ax, do_fdr_correction=False):
    """Performs ANOVA for each time bin and marks significant bins on the plot."""
    num_bins = len(timeline) - 1
    spike_df = pd.DataFrame(spike_data, columns=["spike_counts", "transition"])
    p_values = []

    # Define the time window for counting significant bins
    time_window_start, time_window_end = -0.45, 0.45
    valid_bins = (timeline[:-1] >= time_window_start) & (timeline[:-1] <= time_window_end)

    for bin_idx in range(num_bins):
        try:
            # Extract spike counts per bin (handle cases where spike_counts are not full-length arrays)
            bin_data = spike_df["spike_counts"].dropna().apply(
                lambda x: x[bin_idx] if len(x) > bin_idx else np.nan
            ).dropna()
            groups = spike_df["transition"].loc[bin_data.index]
            unique_groups = groups.unique()
            spike_lists = [bin_data[groups == grp].tolist() for grp in unique_groups]

            if len(spike_lists) > 1:
                _, p_value = f_oneway(*spike_lists)
            else:
                p_value = 1.0  # If not enough groups to compare, return non-significance
        except Exception as e:
            print(f"Error in ANOVA at bin {bin_idx}: {e}")
            p_value = 1.0

        p_values.append(p_value)

    # Multiple comparison correction (if required)
    if do_fdr_correction:
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    else:
        corrected_p_values = p_values

    corrected_p_values = np.array(corrected_p_values)

    # Ensure that valid_bins is applied **before** extracting significant bins
    significant_bins = np.where((corrected_p_values < 0.05) & valid_bins)[0]

    # Mark significant bins on the plot using bin centers
    for bin_idx in significant_bins:
        if bin_idx + 1 < len(timeline):  # Ensure index is within range
            bin_center = (timeline[bin_idx] + timeline[bin_idx + 1]) / 2
            ax.axvline(bin_center, color='red', linestyle='--', alpha=0.5)  # **Corrected vertical line placement**

    return significant_bins  # Return the indices of significant bins



def __plot_region_summary(statistical_summary, spike_times_df, region_summary_dir):
    """Creates a summary plot showing the counts and percentages of significant units per region for each ROI and transition across all sessions."""
    regions = spike_times_df["region"].unique()
    rois = ["eyes", "non_eye_face", "object", "out_of_roi"]
    transitions = ["from", "to"]
    
    total_units_per_region = {region: len(spike_times_df[spike_times_df["region"] == region]) for region in regions}
    significant_counts = np.zeros((len(regions), len(rois), len(transitions)))

    for unit_uuid, brain_region, sig_results in statistical_summary:
        for roi_idx, roi in enumerate(rois):
            for trans_idx, transition_type in enumerate(transitions):
                if roi in sig_results and transition_type in sig_results[roi]:
                    significant_counts[regions.tolist().index(brain_region), roi_idx, trans_idx] += 1

    n_regions = len(regions)
    n_cols = 2
    n_rows = int(np.ceil(n_regions / n_cols))  # Ensure enough rows for all regions

    # Determine the common color scale
    max_percent = (significant_counts / np.array(list(total_units_per_region.values()))[:, None, None] * 100)
    max_percent[np.isnan(max_percent)] = 0  # Handle NaNs where total_units_per_region = 0
    vmin, vmax = 0, max_percent.max()

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axs = axs.flatten()  # Flatten in case n_regions < 4

    for i, region in enumerate(regions):
        total_units = total_units_per_region[region]
        percent_significant = (significant_counts[i] / total_units) * 100 if total_units > 0 else np.zeros_like(significant_counts[i])
        
        # Prepare text overlay with significant unit count and percentage
        table_data = [[f"{int(significant_counts[i, j, k])}\n({percent_significant[j, k]:.1f}%)"  
                       for k in range(len(transitions))] for j in range(len(rois))]
        
        im = axs[i].imshow(percent_significant, cmap="Reds", aspect="auto", vmin=vmin, vmax=vmax)
        axs[i].set_xticks(range(len(transitions)))
        axs[i].set_xticklabels(transitions)
        axs[i].set_yticks(range(len(rois)))
        axs[i].set_yticklabels(rois)
        axs[i].set_title(f"{region} (Total Units: {total_units})")  # Added total unit count to title

        # Add significant unit counts and percentage as text overlay
        for j in range(len(rois)):
            for k in range(len(transitions)):
                axs[i].text(k, j, table_data[j][k], ha="center", va="center", color="black")

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjusted for better positioning
    fig.colorbar(im, cax=cbar_ax, label="Percentage of Significant Units")

    # Remove empty subplots if regions < 4
    for i in range(n_regions, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    plt.savefig(os.path.join(region_summary_dir, "all_region_summary.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()






# ** ARCHIVE **

def __compute_fixation_transition_probabilities(agent_behav_df):
    transitions = []
    for _, run_df in agent_behav_df.groupby("run_number"):
        fixation_sequences = run_df["fixation_location"].values[0]
        categorized_fixations = [
            "eyes" if {"face", "eyes_nf"}.issubset(set(fixes)) else
            "non_eye_face" if "face" in set(fixes) else
            "object" if set(fixes) & {"left_nonsocial_object", "right_nonsocial_object"} else "out_of_roi"
            for fixes in fixation_sequences
        ]
        transitions.extend(zip(categorized_fixations[:-1], categorized_fixations[1:]))
    return pd.DataFrame(transitions, columns=["from", "to"]).value_counts(normalize=True).reset_index(name="probability")


def __plot_transition_matrix(transition_df, session_dir, agent):
    pivot_table = transition_df.pivot(index="from", columns="to", values="probability").fillna(0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".4f", linewidths=0.5)
    plt.title(f"Fixation Transition Probabilities ({agent})")
    plt.xlabel("To Fixation")
    plt.ylabel("From Fixation")
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, f"fixation_transition_matrix_{agent}.png"), dpi=100)
    plt.close()