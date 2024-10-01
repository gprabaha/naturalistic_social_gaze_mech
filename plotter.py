import matplotlib.pyplot as plt
import numpy as np


def plot_agent_behavior(fixation_data, saccade_data, gaze_data, roi_rects, filename, params):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # First subplot: plot fixations
    plot_fixations(axs[0], fixation_data, gaze_data)
    overlay_roi_rects(axs[0], roi_rects)
    # Second subplot: plot saccades
    plot_saccades(axs[1], saccade_data, gaze_data)
    overlay_roi_rects(axs[1], roi_rects)
    # Third subplot: plot both fixations and saccades (without the 30% opacity points)
    plot_combined(axs[2], fixation_data, saccade_data, gaze_data)
    overlay_roi_rects(axs[2], roi_rects)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_fixations(ax, fixation_data, gaze_data):
    # Extract fixation indices and positions
    fixation_indices = fixation_data['fixationindices']
    fixation_points = gaze_data[fixation_indices]
    # Plot fixation points with 30% opacity
    ax.scatter(fixation_points[:, 0], fixation_points[:, 1], c='black', alpha=0.3)
    # Plot mean fixation points with color gradient based on relative time
    mean_fixations = np.mean(fixation_points, axis=1)
    times = np.arange(len(mean_fixations)) / len(mean_fixations)
    ax.scatter(mean_fixations[:, 0], mean_fixations[:, 1], c=times, cmap='cool', alpha=1)


def plot_saccades(ax, saccade_data, gaze_data):
    # Extract saccade indices and positions
    saccade_indices = saccade_data['saccadeindices']
    saccade_points = gaze_data[saccade_indices]
    # Plot saccade points with 30% opacity
    ax.scatter(saccade_points[:, 0], saccade_points[:, 1], c='black', alpha=0.3)
    # Draw arrows for saccades with color gradient based on relative time
    start_points = gaze_data[saccade_indices[:, 0]]
    end_points = gaze_data[saccade_indices[:, 1]]
    times = np.arange(len(start_points)) / len(start_points)
    for i, (start, end) in enumerate(zip(start_points, end_points)):
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
                 head_width=0.05, color=plt.cm.viridis(times[i]))


def plot_combined(ax, fixation_data, saccade_data, gaze_data):
    # Plot mean fixation points
    plot_fixations(ax, fixation_data, gaze_data)
    # Draw saccade arrows
    plot_saccades(ax, saccade_data, gaze_data)


def overlay_roi_rects(ax, roi_rects):
    # Overlay ROI rects (e.g., face, eyes)
    for roi_name, roi_coords in roi_rects.items():
        rect = plt.Rectangle((roi_coords[0], roi_coords[1]), roi_coords[2] - roi_coords[0], 
                             roi_coords[3] - roi_coords[1], fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
