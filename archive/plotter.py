

import pdb

import logging
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
import random


# Set up a logger for this script
logger = logging.getLogger(__name__)


def plot_fixations_and_saccades(nan_removed_gaze_data_df, fixation_df, saccade_df, params):
    """
    Main function to plot fixations and saccades for each session, interaction type, run, and agent (m1 and m2).
    Uses fixation_df and saccade_df directly for start-stop intervals.
    Saves the plots under a new subfolder for each session in the base plot directory.
    Parameters:
    -----------
    nan_removed_gaze_data_df : pd.DataFrame
        DataFrame containing gaze data with position information and roi rects.
    fixation_df : pd.DataFrame
        DataFrame containing fixation start-stop intervals.
    saccade_df : pd.DataFrame
        DataFrame containing saccade start-stop intervals.
    params : dict
        Dictionary containing configuration such as the root data directory and plot saving paths.
    """
    # Get today's date for the directory structure
    today_date = datetime.now().strftime('%Y-%m-%d')
    base_plot_dir = os.path.join(params['root_data_dir'], 'plots', 'fixations_and_saccades', today_date)
    os.makedirs(base_plot_dir, exist_ok=True)
    # Gather tasks across all sessions, interactions, and runs
    tasks = []
    for session in nan_removed_gaze_data_df['session_name'].unique():
        session_dir = os.path.join(base_plot_dir, session)
        os.makedirs(session_dir, exist_ok=True)
        # Filter data for the current session
        session_gaze_data = nan_removed_gaze_data_df[nan_removed_gaze_data_df['session_name'] == session]
        session_fixation_data = fixation_df[fixation_df['session_name'] == session]
        session_saccade_data = saccade_df[saccade_df['session_name'] == session]
        # Collect tasks for all runs and interactions within the session
        for interaction in session_gaze_data['interaction_type'].unique():
            for run in session_gaze_data['run_number'].unique():
                tasks.append((session, interaction, run, session_gaze_data, session_fixation_data, session_saccade_data, session_dir))
    # Plot either in parallel or serial based on the use_parallel flag
    if params.get('use_parallel', False):
        logger.info(f"Plotting across all tasks in parallel using {params['num_cpus']} CPUs.")
        with Pool(processes=params['num_cpus']) as pool:
            list(tqdm(pool.imap(_plot_fix_sac_run_wrapper, tasks), total=len(tasks), desc="Plotting fixation and saccades for runs (Parallel)"))
    else:
        logger.info(f"Plotting all tasks serially.")
        for task in tqdm(tasks, total=len(tasks), desc="Plotting fixation and saccades for runs (Serial)"):
            _plot_fix_sac_for_run(*task)


def _plot_fix_sac_run_wrapper(args):
    """
    Wrapper function to unpack arguments for _plot_for_run for parallel execution.
    """
    return _plot_fix_sac_for_run(*args)


def _plot_fix_sac_for_run(session, interaction, run, session_gaze_data, session_fixation_data, session_saccade_data, session_dir):
    """
    Function to plot fixations and saccades for a specific session, interaction, and run.
    Optimized to gather points before plotting.
    """
    # Filter data for the run
    run_data_gaze = session_gaze_data[
        (session_gaze_data['interaction_type'] == interaction) &
        (session_gaze_data['run_number'] == run)
    ]
    run_fixation_data = session_fixation_data[
        (session_fixation_data['interaction_type'] == interaction) &
        (session_fixation_data['run_number'] == run)
    ]
    run_saccade_data = session_saccade_data[
        (session_saccade_data['interaction_type'] == interaction) &
        (session_saccade_data['run_number'] == run)
    ]
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns (for m1 and m2)
    for agent, col in zip(['m1', 'm2'], [0, 1]):
        agent_gaze_data = run_data_gaze[run_data_gaze['agent'] == agent]
        agent_fixation_data = run_fixation_data[run_fixation_data['agent'] == agent]
        agent_saccade_data = run_saccade_data[run_saccade_data['agent'] == agent]
        if agent_gaze_data.empty or agent_fixation_data.empty or agent_saccade_data.empty:
            continue
        # Gather data for plotting first
        fix_points, fix_means = _gather_fixation_data(agent_gaze_data, agent_fixation_data)
        sac_points, sac_arrows = _gather_saccade_data(agent_gaze_data, agent_saccade_data)
        # Plot fixations
        _plot_fixation_points_and_means(fix_points, fix_means, axs[0, col], f'{agent} - Fixations')
        # Plot saccades
        _plot_saccade_points_and_arrows(sac_points, sac_arrows, axs[1, col], f'{agent} - Saccades')
        # Plot combined fixations and saccades
        _plot_combined(fix_means, sac_arrows, axs[2, col], f'{agent} - Combined Fixations and Saccades')
        # Overlay ROI rectangles
        _overlay_roi_rects_for_run(agent_gaze_data, axs[:, col])
    plt.tight_layout()
    plot_filename = f'{session}_{interaction}_run{run}.png'
    plt.savefig(os.path.join(session_dir, plot_filename))
    plt.close()


def _gather_fixation_data(agent_gaze_data, agent_fixation_data):
    """
    Gather fixation points and their mean positions.
    """
    fixation_intervals = agent_fixation_data['fixation_start_stop'].values[0]
    positions = agent_gaze_data['positions'].values[0]
    fix_points = []
    fix_means = []
    for start, stop in fixation_intervals:
        fix_points.append(positions[start:stop+1])
        mean_x = positions[start:stop+1, 0].mean()
        mean_y = positions[start:stop+1, 1].mean()
        time_color = (start + stop) / (2 * len(positions))  # Normalize time for color
        fix_means.append((mean_x, mean_y, time_color))
    return np.concatenate(fix_points), fix_means


def _gather_saccade_data(agent_gaze_data, agent_saccade_data):
    """
    Gather saccade points and arrows (start to end positions).
    """
    saccade_intervals = agent_saccade_data['saccade_start_stop'].values[0]
    positions = agent_gaze_data['positions'].values[0]
    sac_points = []
    sac_arrows = []
    for start, stop in saccade_intervals:
        sac_points.append(positions[start:stop+1])
        time_color = start / len(positions)  # Normalize time for color
        sac_arrows.append((positions[start], positions[stop], time_color))
    return np.concatenate(sac_points), sac_arrows


def _plot_fixation_points_and_means(points, means, ax, title):
    """
    Plot the gathered fixation points and their mean positions.
    """
    ax.set_title(title)
    # Plot all fixation points at 10% opacity
    ax.scatter(points[:, 0], points[:, 1], color='gray', alpha=0.1)
    # Plot mean fixation points, color-coded by time
    for mean_x, mean_y, time_color in means:
        ax.scatter(mean_x, mean_y, color=cm.viridis(time_color), edgecolor='black', s=100)


def _plot_saccade_points_and_arrows(points, arrows, ax, title):
    """
    Plot the gathered saccade points and arrows.
    """
    ax.set_title(title)
    # Plot all saccade points at 10% opacity
    ax.scatter(points[:, 0], points[:, 1], color='gray', alpha=0.1)
    # Plot arrows for each saccade (start to end), color-coded by time
    for start, stop, time_color in arrows:
        ax.arrow(start[0], start[1], stop[0] - start[0], stop[1] - start[1],
                 color=cm.viridis(time_color), head_width=5, head_length=5)


def _plot_combined(means, arrows, ax, title):
    """
    Plot the combined fixation means and saccade arrows.
    """
    ax.set_title(title)
    # Plot mean fixation points
    for mean_x, mean_y, time_color in means:
        ax.scatter(mean_x, mean_y, color=cm.viridis(time_color), edgecolor='black', s=100)
    # Plot saccade arrows
    for start, stop, time_color in arrows:
        ax.arrow(start[0], start[1], stop[0] - start[0], stop[1] - start[1],
                 color=cm.viridis(time_color), head_width=5, head_length=5)



def _overlay_roi_rects_for_run(agent_gaze_data, axs):
    """
    Overlay ROI rectangles on the subplots for each axis in axs.
    """
    roi_rects = agent_gaze_data['roi_rects'].values[0]
    # Directly add ROI rectangle patches to each subplot
    for ax in axs:
        for roi_name, rect in roi_rects.items():
            x1, y1, x2, y2 = rect
            ax.add_patch(plt.Rectangle((x1, y1),
                                       x2 - x1,
                                       y2 - y1,
                                       edgecolor='red', facecolor='none'))


def plot_random_run_snippets(neural_fr_timeseries_df, snippet_duration=1, bin_width=0.01):
    """
    Plot three random 1-second snippets of firing rate timeseries for a random run in a random session.
    Args:
        neural_fr_timeseries_df (pd.DataFrame): DataFrame with firing rate timeseries for each session, interaction type,
                                                run, region, and unit.
        snippet_duration (int): Duration of each snippet in seconds.
    """
    # Select a random session and run
    random_session = random.choice(neural_fr_timeseries_df['session_name'].unique())
    session_df = neural_fr_timeseries_df[neural_fr_timeseries_df['session_name'] == random_session]
    random_run = random.choice(session_df['run_number'].unique())
    # Filter DataFrame for the selected session and run
    run_df = session_df[session_df['run_number'] == random_run]
    # Calculate the number of bins per snippet
    bins_per_snippet = int(snippet_duration / bin_width)
    sampling_rate = bins_per_snippet / snippet_duration
    # Plot three random 1-second snippets in subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Firing Rate Snippets for Session {random_session}, Run {random_run}")
    for i, ax in enumerate(axes):
        # Select a random start point for each snippet
        max_start = len(run_df.iloc[0]['binned_neural_fr_in_run']) - bins_per_snippet
        start_bin = random.randint(0, max_start)
        end_bin = start_bin + bins_per_snippet
        # Plot each unit's snippet within the chosen range
        for _, row in run_df.iterrows():
            unit_timeseries = row['binned_neural_fr_in_run'][start_bin:end_bin]
            time_points = np.arange(len(unit_timeseries)) / sampling_rate  # Time in seconds
            ax.plot(time_points, unit_timeseries, label=f"{row['region']} - {row['unit_uuid']}", alpha=0.7)
        ax.set_title(f"Snippet {i + 1} (from {start_bin / sampling_rate:.2f} s to {end_bin / sampling_rate:.2f} s)")
        ax.set_ylabel("Firing Rate (spikes/s)")
    # Finalize plot
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plot.png")
    plt.close()
    print(f"Plot saved as plot.png for session {random_session}, run {random_run}")



import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.colors import to_rgba

def process_and_plot_behaviors(binary_behav_timeseries_df, save_dir):
    """
    Process behaviors for all runs and agents, and plot/save them.
    Args:
        binary_behav_timeseries_df (pd.DataFrame): DataFrame with binary behavior timelines and labels.
        save_dir (str): Directory to save the plots.
    """
    # Create a date-based directory under save_dir
    date_dir = os.path.join(save_dir, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    grouped = binary_behav_timeseries_df.groupby(['session_name', 'interaction_type', 'run_number'])
    for (session_name, interaction_type, run_number), group_df in grouped:
        unique_from = group_df["from"].unique()
        unique_to = group_df["to"].unique()

        # Generate behaviors
        behaviors = _generate_unique_behaviors(unique_from, unique_to)

        # Extract binary timelines for both agents
        agent_timelines_m1 = _extract_agent_timelines(group_df, behaviors, "m1")
        agent_timelines_m2 = _extract_agent_timelines(group_df, behaviors, "m2")

        # Create session-specific folder under date_dir
        session_dir = os.path.join(date_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)

        # Plot and save the tiling of behaviors for both agents
        _plot_behavior_tiling(agent_timelines_m1, "m1", session_dir, session_name, interaction_type, run_number)
        _plot_behavior_tiling(agent_timelines_m2, "m2", session_dir, session_name, interaction_type, run_number)


def _generate_unique_behaviors(unique_from, unique_to):
    """
    Generate all possible behaviors for an agent.
    Args:
        unique_from (array-like): Unique 'from' values.
        unique_to (array-like): Unique 'to' values.
    Returns:
        list: List of all possible behaviors for an agent.
    """
    unique_from = [loc for loc in unique_from if loc != "all"]
    unique_to = [loc for loc in unique_to if loc != "all"]

    behaviors = []
    for loc in unique_from:
        behaviors.append(("fixation", loc, loc))
    behaviors.append(("fixation", "any", "any"))

    for to in unique_to:
        behaviors.append(("saccade", "any", to))
    for from_ in unique_from:
        behaviors.append(("saccade", from_, "any"))
    behaviors.append(("saccade", "any", "any"))

    return behaviors


def _extract_agent_timelines(group_df, behaviors, agent):
    """
    Extract binary timelines for all behaviors of a specific agent.
    Args:
        group_df (pd.DataFrame): Grouped DataFrame for a run.
        behaviors (list): List of behaviors to extract.
        agent (str): Agent name ('m1' or 'm2').
    Returns:
        dict: Dictionary with behavior names as keys and binary timelines as values.
    """
    return {
        f"{behav_type}_{from_}_{to}": _extract_binary_vector(behav_type, from_, to, group_df, agent)
        for behav_type, from_, to in behaviors
    }


def _extract_binary_vector(behav_type, behav_from, behav_to, data, agent):
    """
    Extract binary vectors based on behavior type, agent, and 'from'-'to' combinations.
    Args:
        behav_type (str): Behavior type ('fixation' or 'saccade').
        behav_from (str): 'From' field value.
        behav_to (str): 'To' field value.
        data (pd.DataFrame): DataFrame containing behavioral data.
        agent (str): The agent ('m1' or 'm2') whose data to extract.
    Returns:
        np.array or None: Binary vector or None if no matching data.
    """
    # Build the filter condition based on behavior type
    if behav_from == "any" and behav_to == "any":
        # No need for 'from' or 'to' filtering; take all cases for the agent
        filter_cond = (data['behav_type'] == behav_type) & (data['agent'] == agent)
    elif behav_type == "fixation":
        filter_cond = (
            (data['behav_type'] == behav_type) &
            (data['from'] == behav_from) &
            (data['to'] == behav_to) &
            (data['agent'] == agent)
        )
    elif behav_type == "saccade":
        if behav_from == "any":
            filter_cond = (
                (data['behav_type'] == behav_type) &
                (data['to'] == behav_to) &
                (data['agent'] == agent)
            )
        elif behav_to == "any":
            filter_cond = (
                (data['behav_type'] == behav_type) &
                (data['from'] == behav_from) &
                (data['agent'] == agent)
            )
        else:
            filter_cond = (
                (data['behav_type'] == behav_type) &
                (data['from'] == behav_from) &
                (data['to'] == behav_to) &
                (data['agent'] == agent)
            )
    else:
        logger.warning(f"Unsupported behavior type: {behav_type}")
        return None

    # Apply the filter and extract binary timelines
    filtered_timelines = data[filter_cond]['binary_vector'].apply(np.array)
    valid_timelines = [
        arr for arr in filtered_timelines 
        if isinstance(arr, np.ndarray) and arr.ndim == 1
    ]
    # Combine timelines with a bitwise OR if valid timelines exist
    if valid_timelines:
        try:
            combined_timelines = np.bitwise_or.reduce(np.vstack(valid_timelines))
            return combined_timelines
        except ValueError as e:
            logger.error(f"Error combining timelines: {e}")
            logger.debug(f"Valid timelines: {valid_timelines}")
    else:
        logger.debug(
            f"No valid binary timelines found after filtering for: {agent}: {behav_type} from {behav_from} to {behav_to}"
        )
    return None



def _plot_behavior_tiling(agent_timelines, agent, session_dir, session_name, interaction_type, run_number):
    """
    Plot behaviors tiling the timeline with overlaps shown via colors.
    Args:
        agent_timelines (dict): Dictionary with behavior names as keys and binary vectors as values.
        agent (str): Agent name ('m1' or 'm2').
        session_dir (str): Directory to save the plots for the session.
        session_name (str): Session name for subfolder.
        interaction_type (str): Interaction type for the filename.
        run_number (int): Run number for the plot name.
    """
    timeline_length = len(next(iter(agent_timelines.values())))
    behavior_matrix = np.zeros((len(agent_timelines), timeline_length), dtype=int)
    behavior_names = list(agent_timelines.keys())

    # Fill behavior matrix
    for i, (behavior, timeline) in enumerate(agent_timelines.items()):
        if timeline is not None:
            behavior_matrix[i, :] = timeline

    # Assign unique colors to behaviors
    num_behaviors = len(behavior_names)
    color_palette = plt.cm.get_cmap('tab20', num_behaviors)
    behavior_colors = [to_rgba(color_palette(i)) for i in range(num_behaviors)]

    # Combine behavior timelines for tiling
    plt.figure(figsize=(12, 6))
    for i in range(timeline_length):
        active_behaviors = np.where(behavior_matrix[:, i] == 1)[0]
        if len(active_behaviors) == 1:
            # Single behavior active
            plt.plot([i, i + 1], [0, 0], color=behavior_colors[active_behaviors[0]], linewidth=5)
        elif len(active_behaviors) > 1:
            # Multiple behaviors active (blend colors)
            blended_color = np.mean([behavior_colors[j] for j in active_behaviors], axis=0)
            plt.plot([i, i + 1], [0, 0], color=blended_color, linewidth=5)

    # Create legend
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=4, label=behavior)
        for behavior, color in zip(behavior_names, behavior_colors)
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{agent} Behaviors Tiling - Session: {session_name}, Interaction: {interaction_type}, Run: {run_number}")
    plt.xlabel("Time (bins)")
    plt.yticks([])  # Remove y-axis ticks
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    # Save plot with interaction type in the filename
    plot_path = os.path.join(
        session_dir, f"{agent}_behavior_tiling_{interaction_type}_run_{run_number}.png"
    )
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()








# Define the plotting function
def plot_auto_and_cross_correlations(binary_timeseries_scaled_auto_and_crosscorr_df, params):
    """
    Plots the mean autocorrelations and crosscorrelations across all runs
    for unique behav_type-from-to combinations, organized by session.

    Args:
        binary_timeseries_scaled_auto_and_crosscorr_df (pd.DataFrame):
            DataFrame containing autocorrelations and crosscorrelations.
        params (dict):
            Parameters dictionary containing 'root_data_dir'.
    """
    # Get today's date for the folder structure
    today_date = datetime.now().strftime('%Y-%m-%d')
    base_plot_dir = os.path.join(
        params['root_data_dir'], 'plots', 'auto_and_crosscorrelations', today_date
    )
    os.makedirs(base_plot_dir, exist_ok=True)

    # Iterate over each session and interaction type
    for (session_name, interaction_type), session_group in tqdm(
        binary_timeseries_scaled_auto_and_crosscorr_df.groupby([
            'session_name', 'interaction_type'
        ]), desc="Processing sessions"
    ):
        # Create a folder for the session and interaction type
        session_plot_dir = os.path.join(base_plot_dir, session_name, interaction_type)
        os.makedirs(session_plot_dir, exist_ok=True)

        # Find unique behav_type-from-to combinations
        unique_combinations = session_group[['behav_type', 'from', 'to']].drop_duplicates()
        for _, (behav_type, from_, to) in tqdm(
            unique_combinations.iterrows(), 
            total=len(unique_combinations), 
            desc=f"{session_name} - {interaction_type}",
            leave=False
        ):
            # Filter the data for the specific combination
            filtered_data = session_group[
                (session_group['behav_type'] == behav_type) &
                (session_group['from'] == from_) &
                (session_group['to'] == to)
            ]
            # Cut the initial 10 seconds of data (sampled at 1kHz)
            cut_data = {
                'autocorr_m1': [x[:10000] for x in filtered_data['autocorr_m1']],
                'autocorr_m2': [x[:10000] for x in filtered_data['autocorr_m2']],
                'crosscorr_m1_m2': [x[:10000] for x in filtered_data['crosscorr_m1_m2']],
                'crosscorr_m2_m1': [x[:10000] for x in filtered_data['crosscorr_m2_m1']]
            }

            # Compute the mean autocorrelations and crosscorrelations across runs
            mean_autocorr_m1 = np.mean(cut_data['autocorr_m1'], axis=0)
            mean_autocorr_m2 = np.mean(cut_data['autocorr_m2'], axis=0)
            mean_crosscorr_m1_m2 = np.mean(cut_data['crosscorr_m1_m2'], axis=0)
            mean_crosscorr_m2_m1 = np.mean(cut_data['crosscorr_m2_m1'], axis=0)

            # Define the time range (first 10 seconds, sampled at 1kHz)
            time_range = np.arange(0, 10000) / 1000

            # Plot the data
            plt.figure(figsize=(10, 6))

            plt.plot(time_range, mean_autocorr_m1, label='Autocorr M1', alpha=0.7)
            plt.plot(time_range, mean_autocorr_m2, label='Autocorr M2', alpha=0.7)
            plt.plot(time_range, mean_crosscorr_m1_m2, label='Crosscorr M1->M2', alpha=0.7)
            plt.plot(time_range, mean_crosscorr_m2_m1, label='Crosscorr M2->M1', alpha=0.7)

            plt.title(f'{interaction_type}: {behav_type} - From: {from_}, To: {to}')
            plt.xlabel('Time (s)')
            plt.ylabel('Correlation')
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_filename = f'{interaction_type}_{behav_type}_from_{from_}_to_{to}.png'
            plot_filepath = os.path.join(session_plot_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()

            print(f'Plot saved: {plot_filepath}')



def plot_mean_auto_and_crosscorrelations_for_monkey_pairs(recording_sessions_and_monkeys, binary_corr_df, params):
    # Merge the dataframes to include `m1` and `m2` information
    merged_df = binary_corr_df.merge(recording_sessions_and_monkeys, on='session_name')
    # Ensure base plot directory exists
    today_date = datetime.now().strftime('%Y-%m-%d')
    base_plot_dir = os.path.join(
        params['root_data_dir'],
        'plots',
        'mean_auto_and_cross_correlations_across_pairs',
        today_date
    )
    os.makedirs(base_plot_dir, exist_ok=True)
    # Group by `m1`, `m2`, `behav_type`, `from`, and `to`
    grouped = merged_df.groupby(['m1', 'm2', 'behav_type', 'from', 'to'])
    for (m1, m2, behav_type, from_, to_), group in grouped:
        # Create directories for the pair and behavior type
        pair_dir = os.path.join(base_plot_dir, f"{m1}_{m2}")
        behav_dir = os.path.join(pair_dir, behav_type)
        os.makedirs(behav_dir, exist_ok=True)
        # Truncate vectors to 20 seconds (20,000 samples)
        mean_corrs = _truncate_and_average(group, ['autocorr_m1', 'autocorr_m2', 'crosscorr_m1_m2', 'crosscorr_m2_m1'], max_len=20000)
        # Plot and save
        _plot_correlation(mean_corrs, m1, m2, behav_type, from_, to_, behav_dir)


def _truncate_and_average(group, columns, max_len):
    """Truncate vectors to a fixed length and compute their mean."""
    truncated_means = {}
    for col in columns:
        truncated = [vec[:max_len] for vec in group[col] if len(vec) >= max_len]
        truncated_means[col] = np.mean(truncated, axis=0) if truncated else np.zeros(max_len)
    return truncated_means


def _plot_correlation(mean_corrs, m1, m2, behav_type, from_, to_, output_dir):
    """Plot the mean correlations and save to file."""
    plt.figure(figsize=(10, 6))
    plt.title(f"Mean Correlations: {m1}-{m2}, {behav_type}, from {from_} to {to_}")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    lags = np.arange(len(mean_corrs['autocorr_m1']))
    # plt.plot(lags, mean_corrs['autocorr_m1'], label='Autocorr M1', linestyle='--')
    # plt.plot(lags, mean_corrs['autocorr_m2'], label='Autocorr M2', linestyle='--')
    plt.plot(lags, mean_corrs['crosscorr_m1_m2'], label='Crosscorr M1->M2', linestyle='-')
    plt.plot(lags, mean_corrs['crosscorr_m2_m1'], label='Crosscorr M2->M1', linestyle='-')
    plt.legend()
    plt.tight_layout()
    # Save plot
    filename = f"{m1}_{m2}_{behav_type}_from_{from_}_to_{to_}.png".replace(" ", "_")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
