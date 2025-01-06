
import numpy as np
from scipy.interpolate import interp1d

import fixation_detector
import saccade_detector


def detect_fixation_and_saccade_in_run(positions, session_name):
    pdb.set_trace()
    positions = interpolate_nans_in_positions_with_sliding_window(positions)
    pdb.set_trace()
    non_nan_chunks, chunk_start_indices = extract_non_nan_chunks(positions)
    
    pdb.set_trace()

    for position_chunk, start_ind in zip(non_nan_chunks, chunk_start_indices):
        fixation_indices = fixation_detector.detect_fixation_in_position_array(position_chunk, session_name)
        fixation_indices = fixation_indices + start_ind
        saccade_indices = saccade_detector.detect_saccade_in_position_array(position_chunk)
        saccade_indices = saccade_indices + start_ind


def interpolate_nans_in_positions_with_sliding_window(positions, window_size=15, max_nans=5):
    """
    Interpolates NaN values in a sliding window of data if the window contains fewer than max_nans.
    Parameters:
        positions (np.ndarray): 2D array of positions with shape (N, 2).
        window_size (int): Size of the sliding window.
        max_nans (int): Maximum number of NaN values allowed for interpolation within a window.
    Returns:
        np.ndarray: Positions with interpolated values.
    """
    positions = positions.copy()  # Avoid modifying the original array
    num_points = positions.shape[0]
    for start in range(num_points - window_size + 1):
        # Define the window
        end = start + window_size
        window = positions[start:end]
        # Check the number of NaNs in the window
        nan_mask = np.isnan(window).any(axis=1)
        nan_count = np.sum(nan_mask)
        if nan_count <= max_nans:
            # Perform cubic interpolation for each column (x and y)
            for col in range(positions.shape[1]):
                valid_indices = np.where(~np.isnan(window[:, col]))[0]
                if len(valid_indices) > 1:  # Ensure there are enough points to interpolate
                    valid_values = window[valid_indices, col]
                    interp_func = interp1d(
                        valid_indices,
                        valid_values,
                        kind='cubic',
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    # Fill NaN values in the window
                    nan_indices = np.where(nan_mask)[0]
                    interpolated_values = interp_func(nan_indices)
                    window[nan_indices, col] = interpolated_values
            # Update the original positions array with interpolated values
            positions[start:end] = window
    return positions


def extract_non_nan_chunks(positions):
    """
    Extracts continuous chunks of data without NaN values and their start indices.
    Parameters:
        positions (np.ndarray): 2D array of positions with shape (N, 2).
    Returns:
        list: A list of continuous non-NaN chunks, each as a 2D numpy array.
        list: A list of start indices for each chunk.
    """
    non_nan_chunks = []
    start_indices = []
    n = positions.shape[0]
    # Create a mask of rows that have no NaNs
    valid_mask = ~np.isnan(positions).any(axis=1)
    # Find indices where the mask changes from True to False or vice versa
    diff = np.diff(valid_mask.astype(int))
    chunk_starts = np.where(diff == 1)[0] + 1
    chunk_ends = np.where(diff == -1)[0] + 1
    # Handle case where valid_mask starts or ends with True
    if valid_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if valid_mask[-1]:
        chunk_ends = np.append(chunk_ends, n)
    # Extract chunks and their start indices
    for start, end in zip(chunk_starts, chunk_ends):
        non_nan_chunks.append(positions[start:end])
        start_indices.append(start)
    return non_nan_chunks, start_indices
