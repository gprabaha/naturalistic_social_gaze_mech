



import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import pdb

def plot_fixations_and_saccades(nan_removed_gaze_data_df, binary_behav_timeseries_df, params):
    """
    Plots fixations and saccades for each session, interaction type, run, and agent (m1 and m2).
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

    base_plot_dir = os.path.join(params['root_data_dir'], 'plots', 'fixations_and_saccades', params['today_date'])
    os.makedirs(base_plot_dir, exist_ok=True)

    # Loop through each session in the DataFrame
    for session in nan_removed_gaze_data_df['session_name'].unique():
        session_dir = os.path.join(base_plot_dir, session)
        os.makedirs(session_dir, exist_ok=True)

        # Get data for the current session
        session_gaze_data = nan_removed_gaze_data_df[nan_removed_gaze_data_df['session_name'] == session]
        session_behavior_data = binary_behav_timeseries_df[binary_behav_timeseries_df['session_name'] == session]

        # Loop through each interaction type and run
        for interaction in session_gaze_data['interaction_type'].unique():
            for run in session_gaze_data['run_number'].unique():
                run_data_gaze = session_gaze_data[
                    (session_gaze_data['interaction_type'] == interaction) &
                    (session_gaze_data['run_number'] == run)
                ]
                run_data_behavior = session_behavior_data[
                    (session_behavior_data['interaction_type'] == interaction) &
                    (session_behavior_data['run_number'] == run)
                ]

                # Create a plot for each run
                fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns (for m1 and m2)

                for agent, col in zip(['m1', 'm2'], [0, 1]):
                    # Get data for the current agent
                    agent_gaze_data = run_data_gaze[run_data_gaze['agent'] == agent]
                    agent_behavior_data = run_data_behavior[run_data_behavior['agent'] == agent]

                    if agent_gaze_data.empty or agent_behavior_data.empty:
                        continue

                    # Fixation plot (Subplot 1)
                    axs[0, col].set_title(f'{agent} - Fixations')
                    fixations = np.where(agent_behavior_data['fixation_binary_vector'].values[0])[0]
                    positions = agent_gaze_data['positions'].values[0]

                    # Plot all fixation points at 10% opacity
                    axs[0, col].scatter(positions[fixations, 0], positions[fixations, 1], c='gray', alpha=0.1)

                    # Plot the mean position of each fixation, color-coded by time
                    time_colors = np.linspace(0, 1, len(fixations))
                    axs[0, col].scatter(positions[fixations, 0], positions[fixations, 1], c=time_colors, cmap='viridis')

                    # Saccade plot (Subplot 2)
                    axs[1, col].set_title(f'{agent} - Saccades')
                    saccades = np.where(agent_behavior_data['saccade_binary_vector'].values[0])[0]

                    # Plot all saccade points at 10% opacity
                    axs[1, col].scatter(positions[saccades, 0], positions[saccades, 1], c='gray', alpha=0.1)

                    # Plot arrows for each saccade (start to end), color-coded by time
                    for start, end in zip(saccades[:-1], saccades[1:]):
                        axs[1, col].arrow(positions[start, 0], positions[start, 1],
                                          positions[end, 0] - positions[start, 0],
                                          positions[end, 1] - positions[start, 1],
                                          color=cm.viridis(time_colors[start]), head_width=5, head_length=5)

                    # Combined plot (Subplot 3)
                    axs[2, col].set_title(f'{agent} - Combined Fixations and Saccades')

                    # Plot mean fixation points (no full data points)
                    axs[2, col].scatter(positions[fixations, 0], positions[fixations, 1], c=time_colors, cmap='viridis')

                    # Plot saccade arrows
                    for start, end in zip(saccades[:-1], saccades[1:]):
                        axs[2, col].arrow(positions[start, 0], positions[start, 1],
                                          positions[end, 0] - positions[start, 0],
                                          positions[end, 1] - positions[start, 1],
                                          color=cm.viridis(time_colors[start]), head_width=5, head_length=5)

                    # Overlay ROI rects
                    roi_rects = agent_gaze_data['roi_rects'].values[0]
                    for roi_name, rect in roi_rects.items():
                        bottom_left, top_right = rect
                        axs[0, col].add_patch(plt.Rectangle(bottom_left,
                                                            top_right[0] - bottom_left[0],
                                                            top_right[1] - bottom_left[1],
                                                            edgecolor='red', facecolor='none'))
                        axs[1, col].add_patch(plt.Rectangle(bottom_left,
                                                            top_right[0] - bottom_left[0],
                                                            top_right[1] - bottom_left[1],
                                                            edgecolor='red', facecolor='none'))
                        axs[2, col].add_patch(plt.Rectangle(bottom_left,
                                                            top_right[0] - bottom_left[0],
                                                            top_right[1] - bottom_left[1],
                                                            edgecolor='red', facecolor='none'))

                plt.tight_layout()
                # Save plot to the session folder
                plot_filename = f'{session}_{interaction}_run{run}.png'
                plt.savefig(os.path.join(session_dir, plot_filename))
                plt.close()
















def gather_plotting_tasks(fixation_dict, saccade_dict, nan_removed_gaze_data_dict, base_plot_dir, params):
    tasks = []
    for session, session_data in fixation_dict.items():
        # Create a separate folder for each session under the date folder
        session_plot_dir = os.path.join(base_plot_dir, session)
        os.makedirs(session_plot_dir, exist_ok=True)
        # Loop through interaction types (e.g., 'interactive' or 'non_interactive')
        for interaction_type, interaction_data in session_data.items():
            if interaction_type == 'non_interactive':
                continue
            for run, run_data in interaction_data.items():
                for agent, _ in run_data.items():
                    # Fetch fixation and saccade data for the agent
                    fixation_data = fixation_dict[session][interaction_type][run].get(agent, None)
                    saccade_data = saccade_dict[session][interaction_type][run].get(agent, None)
                    nan_removed_data = nan_removed_gaze_data_dict[session][interaction_type][run]['positions'][agent]
                    # print(session, interaction_type, run, agent)
                    roi_rects = nan_removed_gaze_data_dict[session][interaction_type][run]['roi_rects'][agent]
                    # Check if there is data for this agent
                    if fixation_data and saccade_data:
                        plot_filename = os.path.join(session_plot_dir, f'{agent}_run_{run}_{interaction_type}.png')
                        tasks.append((fixation_data, saccade_data, nan_removed_data, roi_rects, plot_filename, params))    
    return tasks


def plot_agent_behavior(fixation_data, saccade_data, gaze_data, roi_rects, filename, params):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # First subplot: plot fixations
    plot_fixations(axs[0], fixation_data, gaze_data)
    overlay_roi_rects(axs[0], roi_rects)
    # Second subplot: plot saccades
    plot_saccades(axs[1], saccade_data, gaze_data)
    overlay_roi_rects(axs[1], roi_rects)
    # Third subplot: plot both fixations and saccades (without 30% opacity scatter points)
    plot_combined(axs[2], fixation_data, saccade_data, gaze_data)
    overlay_roi_rects(axs[2], roi_rects)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_fixations(ax, fixation_data, gaze_data):
    # Extract fixation start and stop indices (fixation indices are 2xN, transpose to Nx2)
    fixation_indices = fixation_data['fixationindices'].T
    # Gaze data is already 2xN, so we don't transpose it here
    all_fixation_points = np.hstack([gaze_data[:, start:end] for start, end in fixation_indices])
    mean_fixation_points = np.array([np.mean(gaze_data[:, start:end], axis=1) for start, end in fixation_indices])
    # Calculate relative time for color
    relative_times = np.array([start / gaze_data.shape[1] for start, _ in fixation_indices])
    # Plot all fixation points with 30% opacity
    ax.scatter(all_fixation_points[0, :], all_fixation_points[1, :], c='black', alpha=0.3)
    # Plot mean fixation points with color gradient
    ax.scatter(mean_fixation_points[:, 0], mean_fixation_points[:, 1], c=relative_times, cmap='cool', s=100, alpha=1)


def plot_saccades(ax, saccade_data, gaze_data):
    # Extract saccade start and stop indices (saccade indices are Nx2)
    saccade_indices = saccade_data['saccadeindices']
    # Prepare start and end points of all saccades (use the first and last points in the saccade range)
    start_points = np.array([gaze_data[:, start] for start, _ in saccade_indices])
    end_points = np.array([gaze_data[:, end - 1] for _, end in saccade_indices])
    # Calculate relative time for color
    relative_times = np.array([start / gaze_data.shape[1] for start, _ in saccade_indices])
    # Convert scalar relative_times to RGBA colors using colormap
    colors = plt.cm.viridis(relative_times)
    # Plot all saccade points with 30% opacity
    all_saccade_points = np.hstack([gaze_data[:, start:end] for start, end in saccade_indices])
    ax.scatter(all_saccade_points[0, :], all_saccade_points[1, :], c='black', alpha=0.3)
    # Plot arrows from start to end of saccades with color gradient
    for i in range(len(start_points)):
        ax.arrow(start_points[i][0], start_points[i][1], 
                 end_points[i][0] - start_points[i][0], 
                 end_points[i][1] - start_points[i][1], 
                 head_width=0.05, color=colors[i])  # Pass RGBA color


def plot_combined(ax, fixation_data, saccade_data, gaze_data):
    # Plot mean fixation points (no scatter points)
    fixation_indices = fixation_data['fixationindices'].T
    mean_fixation_points = np.array([np.mean(gaze_data[:, start:end], axis=1) for start, end in fixation_indices])
    relative_times_fix = np.array([start / gaze_data.shape[1] for start, _ in fixation_indices])
    # Convert scalar relative_times to RGBA colors using colormap
    fix_colors = plt.cm.cool(relative_times_fix)
    ax.scatter(mean_fixation_points[:, 0], mean_fixation_points[:, 1], c=fix_colors, s=100, alpha=1)
    # Plot saccade arrows (no scatter points)
    saccade_indices = saccade_data['saccadeindices']
    start_points = np.array([gaze_data[:, start] for start, _ in saccade_indices])
    end_points = np.array([gaze_data[:, end - 1] for _, end in saccade_indices])
    relative_times_sacc = np.array([start / gaze_data.shape[1] for start, _ in saccade_indices])
    # Convert scalar relative_times to RGBA colors using colormap
    sacc_colors = plt.cm.viridis(relative_times_sacc)
    for i in range(len(start_points)):
        ax.arrow(start_points[i][0], start_points[i][1], 
                 end_points[i][0] - start_points[i][0], 
                 end_points[i][1] - start_points[i][1], 
                 head_width=0.05, color=sacc_colors[i])  # Pass RGBA color


def overlay_roi_rects(ax, roi_rects):
    # Overlay ROI rects with bottom-left and top-right corners
    for roi_name, roi_coords in roi_rects.items():
        # roi_coords should have [x_min, y_min, x_max, y_max]
        rect = plt.Rectangle((roi_coords[0], roi_coords[1]), 
                             roi_coords[2] - roi_coords[0], 
                             roi_coords[3] - roi_coords[1], 
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

