

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

def plot_fixations_and_saccades(nan_removed_gaze_data_df, binary_behav_timeseries_df, params):
    """
    Main function to plot fixations and saccades for each session, interaction type, run, and agent (m1 and m2).
    Saves the plots under a new subfolder for each session in the base plot directory.
    Parameters:
    -----------
    nan_removed_gaze_data_df : pd.DataFrame
        DataFrame containing gaze data with position information and roi rects.
    binary_behav_timeseries_df : pd.DataFrame
        DataFrame containing binary fixation and saccade vectors.
    params : dict
        Dictionary containing configuration such as the root data directory and plot saving paths.
    """
    # Get today's date for the directory structure
    today_date = datetime.now().strftime('%Y-%m-%d')
    base_plot_dir = os.path.join(params['root_data_dir'], 'plots', 'fixations_and_saccades', today_date)
    os.makedirs(base_plot_dir, exist_ok=True)
    # Loop through each session in the DataFrame
    for session in nan_removed_gaze_data_df['session_name'].unique():
        session_dir = os.path.join(base_plot_dir, session)
        os.makedirs(session_dir, exist_ok=True)
        # Filter data for the current session
        session_gaze_data = nan_removed_gaze_data_df[nan_removed_gaze_data_df['session_name'] == session]
        session_behavior_data = binary_behav_timeseries_df[binary_behav_timeseries_df['session_name'] == session]
        # Gather the list of runs and interactions
        tasks = []
        for interaction in session_gaze_data['interaction_type'].unique():
            for run in session_gaze_data['run_number'].unique():
                tasks.append((session, interaction, run, session_gaze_data, session_behavior_data, session_dir))
        # Plot either in parallel or serial based on the use_parallel flag
        if params.get('use_parallel', False):
            logger.info(f"Plotting in parallel using {params['num_cpus']} CPUs.")
            with Pool(processes=params['num_cpus']) as pool:
                list(tqdm(pool.imap(_plot_fix_sac_run_wrapper, tasks), total=len(tasks), desc=f"Plotting {session} runs (Parallel)"))
        else:
            logger.info(f"Plotting serially.")
            for task in tqdm(tasks, total=len(tasks), desc=f"Plotting {session} runs (Serial)"):
                _plot_fix_sac_for_run(*task)


def _plot_fix_sac_run_wrapper(args):
    """
    Wrapper function to unpack arguments for _plot_for_run for parallel execution.
    """
    return _plot_fix_sac_for_run(*args)


def _plot_fix_sac_for_run(session, interaction, run, session_gaze_data, session_behavior_data, session_dir):
    """
    Function to plot fixations and saccades for a specific session, interaction, and run.
    Parameters:
    -----------
    session : str - The session name.
    interaction : str - The interaction type.
    run : int - The run number.
    session_gaze_data : pd.DataFrame - Gaze data for the session.
    session_behavior_data : pd.DataFrame - Behavior data for the session.
    session_dir : str - Directory to save the plots.
    """
    # Filter data for the run
    run_data_gaze = session_gaze_data[
        (session_gaze_data['interaction_type'] == interaction) &
        (session_gaze_data['run_number'] == run)
    ]
    run_data_behavior = session_behavior_data[
        (session_behavior_data['interaction_type'] == interaction) &
        (session_behavior_data['run_number'] == run)
    ]
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns (for m1 and m2)
    for agent, col in zip(['m1', 'm2'], [0, 1]):
        agent_gaze_data = run_data_gaze[run_data_gaze['agent'] == agent]
        agent_behavior_data = run_data_behavior[run_data_behavior['agent'] == agent]
        if agent_gaze_data.empty or agent_behavior_data.empty:
            continue
        _plot_fixations_for_run(agent_gaze_data, agent_behavior_data, axs[0, col], agent)
        _plot_saccades_for_run(agent_gaze_data, agent_behavior_data, axs[1, col], agent)
        _plot_combined_fixations_and_saccades_for_run(agent_gaze_data, agent_behavior_data, axs[2, col], agent)
        _overlay_roi_rects_for_run(agent_gaze_data, axs[:, col])
    plt.tight_layout()
    plot_filename = f'{session}_{interaction}_run{run}.png'
    plt.savefig(os.path.join(session_dir, plot_filename))
    plt.close()


def _plot_fixations_for_run(agent_gaze_data, agent_behavior_data, ax, agent):
    """
    Function to plot fixation data.
    """
    ax.set_title(f'{agent} - Fixations')
    fixations = np.where(agent_behavior_data['fixation_binary_vector'].values[0])[0]
    positions = agent_gaze_data['positions'].values[0]
    # Plot all fixation points at 10% opacity
    ax.scatter(positions[fixations, 0], positions[fixations, 1], c='gray', alpha=0.1)
    # Plot the mean position of each fixation, color-coded by time
    time_colors = np.linspace(0, 1, len(fixations))
    ax.scatter(positions[fixations, 0], positions[fixations, 1], c=time_colors, cmap='viridis')


def _plot_saccades_for_run(agent_gaze_data, agent_behavior_data, ax, agent):
    """
    Function to plot saccade data.
    """
    ax.set_title(f'{agent} - Saccades')
    saccades = np.where(agent_behavior_data['saccade_binary_vector'].values[0])[0]
    positions = agent_gaze_data['positions'].values[0]
    # Plot all saccade points at 10% opacity
    ax.scatter(positions[saccades, 0], positions[saccades, 1], c='gray', alpha=0.1)
    # Plot arrows for each saccade (start to end), color-coded by time
    time_colors = np.linspace(0, 1, len(saccades))
    for start, end in zip(saccades[:-1], saccades[1:]):
        ax.arrow(positions[start, 0], positions[start, 1],
                 positions[end, 0] - positions[start, 0],
                 positions[end, 1] - positions[start, 1],
                 color=cm.viridis(time_colors[start]), head_width=5, head_length=5)


def _plot_combined_fixations_and_saccades_for_run(agent_gaze_data, agent_behavior_data, ax, agent):
    """
    Function to plot combined fixation and saccade data.
    """
    ax.set_title(f'{agent} - Combined Fixations and Saccades')
    fixations = np.where(agent_behavior_data['fixation_binary_vector'].values[0])[0]
    saccades = np.where(agent_behavior_data['saccade_binary_vector'].values[0])[0]
    positions = agent_gaze_data['positions'].values[0]
    # Plot mean fixation points
    time_colors = np.linspace(0, 1, len(fixations))
    ax.scatter(positions[fixations, 0], positions[fixations, 1], c=time_colors, cmap='viridis')
    # Plot saccade arrows
    for start, end in zip(saccades[:-1], saccades[1:]):
        ax.arrow(positions[start, 0], positions[start, 1],
                 positions[end, 0] - positions[start, 0],
                 positions[end, 1] - positions[start, 1],
                 color=cm.viridis(time_colors[start]), head_width=5, head_length=5)


def _overlay_roi_rects_for_run(agent_gaze_data, axs):
    """
    Function to overlay ROI rectangles on the subplots.
    """
    roi_rects = agent_gaze_data['roi_rects'].values[0]
    for ax in axs:
        for roi_name, rect in roi_rects.items():
            bottom_left, top_right = rect
            ax.add_patch(plt.Rectangle(bottom_left,
                                       top_right[0] - bottom_left[0],
                                       top_right[1] - bottom_left[1],
                                       edgecolor='red', facecolor='none'))



