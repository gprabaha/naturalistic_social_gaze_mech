import numpy as np
from scipy.interpolate import interp1d


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
