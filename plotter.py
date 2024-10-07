

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




