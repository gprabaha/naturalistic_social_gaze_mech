
import matplotlib.pyplot as plt
import numpy as np
import os

import pdb


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
    # Plot all saccade points with 30% opacity
    all_saccade_points = np.hstack([gaze_data[:, start:end] for start, end in saccade_indices])
    ax.scatter(all_saccade_points[0, :], all_saccade_points[1, :], c='black', alpha=0.3)
    # Plot arrows from start to end of saccades with color gradient
    for i in range(len(start_points)):
        # Use the full RGBA color value
        color = plt.cm.viridis(relative_times[i])
        ax.arrow(start_points[i][0], start_points[i][1], 
                 end_points[i][0] - start_points[i][0], 
                 end_points[i][1] - start_points[i][1], 
                 head_width=0.05, color=color)


def plot_combined(ax, fixation_data, saccade_data, gaze_data):
    # Plot mean fixation points (no scatter points)
    fixation_indices = fixation_data['fixationindices'].T
    mean_fixation_points = np.array([np.mean(gaze_data[:, start:end], axis=1) for start, end in fixation_indices])
    relative_times_fix = np.array([start / gaze_data.shape[1] for start, _ in fixation_indices])
    ax.scatter(mean_fixation_points[:, 0], mean_fixation_points[:, 1], c=relative_times_fix, cmap='cool', s=100, alpha=1)
    # Plot saccade arrows (no scatter points)
    saccade_indices = saccade_data['saccadeindices']
    start_points = np.array([gaze_data[:, start] for start, _ in saccade_indices])
    end_points = np.array([gaze_data[:, end - 1] for _, end in saccade_indices])
    relative_times_sacc = np.array([start / gaze_data.shape[1] for start, _ in saccade_indices])
    for i in range(len(start_points)):
        # Use the full RGBA color value
        color = plt.cm.viridis(relative_times_sacc[i])
        ax.arrow(start_points[i][0], start_points[i][1], 
                 end_points[i][0] - start_points[i][0], 
                 end_points[i][1] - start_points[i][1], 
                 head_width=0.05, color=color)




def overlay_roi_rects(ax, roi_rects):
    # Overlay ROI rects with bottom-left and top-right corners
    for roi_name, roi_coords in roi_rects.items():
        # roi_coords should have [x_min, y_min, x_max, y_max]
        rect = plt.Rectangle((roi_coords[0], roi_coords[1]), 
                             roi_coords[2] - roi_coords[0], 
                             roi_coords[3] - roi_coords[1], 
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

