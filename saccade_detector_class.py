from tqdm import tqdm
import logging
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import pdb


import defaults


class SaccadeDetector:
    def __init__(self, session_name = None, samprate=1/1000, params=None, num_cpus=1):
        self.setup_logger()
        self.session_name = session_name
        self.params = params
        self.samprate = samprate
        self.num_cpus = num_cpus
        self.use_parallel = params['use_parallel']
        self.variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
        self.fltord = 60
        self.lowpasfrq = 30
        self.nyqfrq = 1000 / 2
        self.flt = signal.firwin2(self.fltord, [0, self.lowpasfrq / self.nyqfrq, self.lowpasfrq / self.nyqfrq, 1], [1, 1, 0, 0])
        self.buffer = int(100 / (self.samprate * 1000))
        self.monitor_info = defaults.fetch_monitor_info()


    def setup_logger(self):
        """Sets up the logger for the class."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)


    def detect_saccades_with_edge_outliers(self, data, method="hubel_2000", threshold=None):
        """Detects saccades and microsaccades using different methods and removes edge outliers based on centroid proximity.
        Args:
            data (tuple): Tuple containing x and y coordinates of eye position.
            method (str, optional): Method to use for saccade detection. Can be 'hubel_2000', 'mayo_2023', or None.
                                    Defaults to "hubel_2000".
            threshold (float, optional): Threshold for proximity to max/min values to be considered an outlier.
                                        If not provided, fetch from 'pixel_threshold_for_boundary_outlier_removal'.
            Returns:
                dict: Indices and time points of saccades and microsaccades with start and end indices, 
                    with outliers removed based on centroid proximity to boundaries.
        """
        self.logger.info(f"Detecting saccades using method: {method}")
        # Detect saccades based on the specified method
        if method == "hubel_2000":
            result = self._detect_saccades_hubel_2000(data)
        elif method == "mayo_2023":
            result = self._detect_saccades_mayo_2023(data)
        else:
            result = self._detect_saccades_old_method(data)
        result['session_name'] = self.session_name
        # Extract the coordinates
        x, y = result['XY'][0], result['XY'][1]
        # Fetch pixel threshold for boundary outlier removal
        if not threshold:
            threshold = self.params.get('pixel_threshold_for_boundary_outlier_removal', 50)
        self.logger.info(f"Using pixel threshold for boundary outlier removal: {threshold}")
        # Filter the detected saccades and microsaccades by removing outliers based on centroid proximity
        filtered_saccade_inds = self._filter_out_saccades_by_centroid(result['saccadeindices'], x, y, threshold=threshold)
        filtered_microsaccade_inds = self._filter_out_saccades_by_centroid(result['microsaccadeindices'], x, y, threshold=threshold)
        filtered_saccade_times = filtered_saccade_inds * self.samprate
        filtered_microsaccade_times = filtered_microsaccade_inds * self.samprate
        # Track the removed outliers (saccades and microsaccades that were filtered out due to centroid proximity to boundary)
        removed_saccade_outliers = self._track_outlier_saccades(result['saccadeindices'], filtered_saccade_inds, x, y)
        removed_microsaccade_outliers = self._track_outlier_saccades(result['microsaccadeindices'], filtered_microsaccade_inds, x, y)
        # Store the outlier information separately for saccades and microsaccades
        result['outlier_info'] = {
            'saccades': removed_saccade_outliers,
            'microsaccades': removed_microsaccade_outliers
        }
        # Update the result to include the filtered saccades and microsaccades
        result['saccadeindices'] = filtered_saccade_inds
        result['microsaccadeindices'] = filtered_microsaccade_inds
        result['saccadetimes'] = filtered_saccade_times
        result['microsaccadetimes'] = filtered_microsaccade_times
        # Log the number of saccades and microsaccades finally detected
        num_saccades = filtered_saccade_inds.shape[0] if filtered_saccade_inds.size > 0 else 0
        num_microsaccades = filtered_microsaccade_inds.shape[0] if filtered_microsaccade_inds.size > 0 else 0
        num_saccade_outliers = len(removed_saccade_outliers['start_stop_indices']) if 'start_stop_indices' in removed_saccade_outliers else 0
        num_microsaccade_outliers = len(removed_microsaccade_outliers['start_stop_indices']) if 'start_stop_indices' in removed_microsaccade_outliers else 0
        self.logger.info(f"Final number of saccades detected: {num_saccades}")
        self.logger.info(f"Number of saccade outliers detected: {num_saccade_outliers}")
        self.logger.info(f"Final number of microsaccades detected: {num_microsaccades}")
        self.logger.info(f"Number of microsaccade outliers detected: {num_microsaccade_outliers}")
        return result



    def _detect_saccades_hubel_2000(self, data):
        """Detects saccades and microsaccades using the method inspired by Martinez-Conde, Macknik, and Hubel (2000).
        The algorithm applies a velocity threshold and a direction change criterion to determine eye movement events. 
        This method is informed by the importance of identifying fine-grained eye movements such as microsaccades, 
        which play a crucial role in visual perception and attention, as highlighted in the 2013 Nature Reviews Neuroscience article.
        Args:
            data (tuple): Tuple containing x and y coordinates of eye position in pixels.
        Returns:
            dict: Indices and time points of saccades and microsaccades with start and end indices.
        """
        self.logger.info("Using Hubel et al. (2000) method for saccade detection")
        x, y = self.preprocess_data_old_method(data)
        # Convert x, y from pixels to degrees
        x_deg, y_deg = self.pixels_to_degrees(x, y)
        dx = np.diff(x_deg)
        dy = np.diff(y_deg)
        velocity = np.sqrt(dx**2 + dy**2) / self.samprate  # Convert to degrees per second
        smoothed_velocity = uniform_filter1d(velocity, size=31)
        theta = np.arctan2(dy, dx)
        eye_stopped = np.zeros(len(smoothed_velocity), dtype=bool)
        # Initialize saccades and microsaccades as empty lists
        saccades = []
        microsaccades = []
        for i in tqdm(range(1, len(smoothed_velocity)), desc="Processing velocity data"):
            if smoothed_velocity[i] < 3:  # 3° per second threshold
                eye_stopped[i] = True
            elif np.abs(np.degrees(theta[i] - theta[i-1])) > 15:
                eye_stopped[i] = True
        current_start = None
        for i in range(1, len(eye_stopped)):
            if eye_stopped[i] and not eye_stopped[i-1]:
                if current_start is not None:
                    total_displacement = np.sqrt((x_deg[i] - x_deg[current_start])**2 + (y_deg[i] - y_deg[current_start])**2)
                    if (3/60) <= total_displacement <= 2:
                        microsaccades.append([current_start, i])
                    elif total_displacement > 2:
                        saccades.append([current_start, i])
                    current_start = None
            elif not eye_stopped[i] and eye_stopped[i-1]:
                current_start = i
        # Convert the lists of saccades and microsaccades to NumPy arrays
        saccades = np.array(saccades)
        microsaccades = np.array(microsaccades)
        self.logger.info(f"Detected {len(saccades)} saccades and {len(microsaccades)} microsaccades")
        return {
            'saccadeindices': saccades,
            'microsaccadeindices': microsaccades,
            'XY': np.array([x, y]),
            'variables': self.variables
        }


    def pixels_to_degrees(self, x, y):
        """Converts pixel coordinates to degrees of visual angle, accounting for monitor size and viewing distance.
        Args:
            x (np.ndarray): Array of x-coordinates in pixels.
            y (np.ndarray): Array of y-coordinates in pixels.
        Returns:
            tuple: Two numpy arrays representing x and y coordinates in degrees of visual angle.
        """
        # Monitor specifications
        monitor_diagonal_inches = self.monitor_info['diagonal']  # inches
        monitor_distance_cm = self.monitor_info['distance']  # cm
        vertical_res = self.monitor_info['vertical_resolution']  # pixels
        horizontal_res = self.monitor_info['horizontal_resolution']  # pixels
        # Calculate aspect ratio
        aspect_ratio = horizontal_res / vertical_res
        # Calculate width and height of the monitor in inches
        monitor_height_inches = monitor_diagonal_inches / np.sqrt(1 + aspect_ratio**2)
        monitor_width_inches = monitor_height_inches * aspect_ratio
        # Convert dimensions to cm
        monitor_height_cm = monitor_height_inches * 2.54
        monitor_width_cm = monitor_width_inches * 2.54
        # Calculate the size of one pixel in cm
        pixel_size_cm_x = monitor_width_cm / horizontal_res
        pixel_size_cm_y = monitor_height_cm / vertical_res
        # Convert pixel distances to degrees using the visual angle formula
        x_deg = np.degrees(np.arctan2(x * pixel_size_cm_x, monitor_distance_cm))
        y_deg = np.degrees(np.arctan2(y * pixel_size_cm_y, monitor_distance_cm))
        return x_deg, y_deg


    def _detect_saccades_mayo_2023(self, data):
        """Detects saccades and microsaccades using the method from Willett and Mayo (2023)."""
        self.logger.info("Using Willett and Mayo (2023) method for saccade detection")
        x, y = self.preprocess_data_old_method(data)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        radial_velocity = np.diff(r) * self.samprate
        radial_velocity = np.append(radial_velocity, radial_velocity[-1])
        smoothed_velocity = uniform_filter1d(np.abs(radial_velocity), size=3)
        eye_movement_removed = smoothed_velocity[smoothed_velocity < 6]
        mean_velocity = np.mean(eye_movement_removed)
        std_velocity = np.std(eye_movement_removed)
        onset_threshold = mean_velocity + std_velocity
        saccade_indices, _ = find_peaks(smoothed_velocity, height=6)
        saccade_events = []
        for idx in tqdm(saccade_indices, desc="Processing saccade peaks"):
            start = idx
            while start > 0 and smoothed_velocity[start] > onset_threshold:
                start -= 1
            end = idx
            while end < len(smoothed_velocity) and smoothed_velocity[end] > onset_threshold:
                end += 1
            saccade_events.append([start, end])
        saccades = []
        microsaccades = []
        for start, end in saccade_events:
            total_displacement = np.abs(r[end] - r[start])
            path_length = np.sum(np.sqrt(np.diff(r[start:end+1])**2 + np.diff(theta[start:end+1])**2))
            exclusion_metric = np.log(total_displacement / path_length)
            if exclusion_metric > 1:
                continue
            if total_displacement < 2:
                microsaccades.append([start, end])
            else:
                saccades.append([start, end])
        self.logger.info(f"Detected {len(saccades)} saccades and {len(microsaccades)} microsaccades using Mayo 2023")
        return {
            'saccadeindices': np.array(saccades),
            'microsaccadeindices': np.array(microsaccades),
            'XY': np.array([x, y]),
            'variables': self.variables
        }


    def _detect_saccades_old_method(self, data):
        """Detects saccades and microsaccades using the old method based on velocity thresholding."""
        self.logger.info("Using old method for saccade detection based on velocity thresholds")
        x, y = self.preprocess_data_old_method(data)
        dx = np.diff(x) * self.samprate
        dy = np.diff(y) * self.samprate
        vel = np.sqrt(dx**2 + dy**2)
        # Compute velocity thresholds
        saccade_threshold, micro_saccade_threshold = self.compute_velocity_thresholds_old_method(vel)
        # Detect saccades and micro-saccades based on these thresholds
        saccade_indices = np.where(vel > saccade_threshold)[0]
        micro_saccade_indices = np.where((vel > micro_saccade_threshold) & (vel <= saccade_threshold))[0]
        saccade_event_indices = self.find_event_bounds_old_method(saccade_indices)
        micro_saccade_event_indices = self.find_event_bounds_old_method(micro_saccade_indices)
        self.logger.info(f"Detected {saccade_event_indices.shape[1]} saccades and {micro_saccade_event_indices.shape[1]} microsaccades using old method")
        return {
            'saccadeindices': saccade_event_indices,
            'microsaccadeindices': micro_saccade_event_indices,
            'XY': np.array([x, y]),
            'variables': self.variables
        }


    def preprocess_data_old_method(self, data):
        """Pads, resamples, and filters the eye data.
        Args:
            data (tuple): Tuple containing x and y coordinates of eye data.
        Returns:
            tuple: Preprocessed x and y coordinates.
        """
        self.logger.debug("Preprocessing x and y data using old method")
        x, y = data
        x = np.pad(x, (self.buffer, self.buffer), 'reflect')
        y = np.pad(y, (self.buffer, self.buffer), 'reflect')
        x = self.resample_data(x)
        y = self.resample_data(y)
        x = self.apply_filter(x)
        y = self.apply_filter(y)
        x = x[self.buffer:-self.buffer]
        y = y[self.buffer:-self.buffer]
        return x, y


    def resample_data(self, data):
        """Resamples the data based on the sampling rate.
        Args:
            data (np.ndarray): Array of eye data.
        Returns:
            np.ndarray: Resampled data.
        """
        self.logger.debug("Resampling data")
        t_old = np.linspace(0, len(data) - 1, len(data))
        resample_factor = self.samprate * 1000
        if resample_factor > 1:
            self.logger.error(f"Resample factor is too large: {resample_factor}")
            raise ValueError("Resample factor is too large, leading to excessive memory usage.")
        t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
        f = interp1d(t_old, data, kind='linear')
        return f(t_new)


    def apply_filter(self, data):
        """Applies a low-pass filter to the data.
        Args:
            data (np.ndarray): Array of data to be filtered.
        Returns:
            np.ndarray: Filtered data.
        """
        self.logger.debug("Applying filter to data")
        return signal.filtfilt(self.flt, 1, data)


    def compute_velocity_thresholds_old_method(self, vel_deg):
        """Computes the velocity thresholds for detecting saccades and micro-saccades.
        Args:
            vel_deg (np.ndarray): Array of velocity values in degrees of visual angle per second.
        Returns:
            tuple: Saccade and microsaccade thresholds.
        """
        self.logger.debug("Computing velocity thresholds based on standard deviation for old method")
        vel_mean = np.mean(vel_deg)
        vel_std = np.std(vel_deg)
        saccade_threshold = vel_mean + 6 * vel_std  # Example: 6 * std for saccades
        micro_saccade_threshold = vel_mean + 3 * vel_std  # Example: 3 * std for microsaccades
        self.logger.debug(f"Saccade threshold: {saccade_threshold}, Microsaccade threshold: {micro_saccade_threshold}")
        return saccade_threshold, micro_saccade_threshold


    def find_event_bounds_old_method(self, indices):
        """Finds the start and end indices of events based on threshold crossing.
        Args:
            indices (np.ndarray): Array of indices where events exceed a threshold.
        Returns:
            np.ndarray: Array of start and end indices for each event.
        """
        if len(indices) == 0:
            return np.array([], dtype=int).reshape(2, 0)
        start_inds = [indices[0]]
        end_inds = []
        for i in range(1, len(indices)):
            if indices[i] > indices[i - 1] + 1:
                end_inds.append(indices[i - 1])
                start_inds.append(indices[i])
        end_inds.append(indices[-1])
        self.logger.debug(f"Found {len(start_inds)} events based on threshold crossing")
        return np.array([start_inds, end_inds])
    

    def _filter_out_saccades_by_centroid(self, saccade_indices, x, y, threshold=50):
        """
        Filters out saccades based on the centroid distance from the boundaries.
        A saccade is accepted if its centroid is far enough from the boundary and if more than
        `valid_point_tolerance` (default 90%) of its points are valid.
        Args:
            saccade_indices (np.ndarray): Detected saccades or microsaccades, shape (N, 2) where each row is [start, end].
            x (np.ndarray): x-coordinates corresponding to the original eye data.
            y (np.ndarray): y-coordinates corresponding to the original eye data.
            threshold (float): Px threshold for proximity to max/min values to be considered an outlier.
            valid_point_tolerance (float): Minimum proportion of valid points required to accept a saccade. Defaults to 0.9 (90%).
        Returns:
            np.ndarray: Filtered saccades with shape (M, 2), where M ≤ N.
        """
        filtered_saccades = []
        # Find the min and max values for x and y
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        for saccade in saccade_indices:
            start, end = saccade
            saccade_length = end - start + 1
            if saccade_length <= 0:
                self.logger.debug(f"Skipped invalid saccade with start {start} and end {end}")
                continue  # Skip invalid saccades
            # Extract the points within the saccade
            x_points = x[start:end+1]
            y_points = y[start:end+1]
            # Calculate the centroid of the points
            centroid_x = np.mean(x_points)
            centroid_y = np.mean(y_points)
            # Check if the centroid is far enough from the boundaries
            if (centroid_x < (x_min + threshold)) or (centroid_x > (x_max - threshold)) or \
            (centroid_y < (y_min + threshold)) or (centroid_y > (y_max - threshold)):
                continue  # Discard the saccade if centroid is too close to the boundary
            else:
                filtered_saccades.append([start, end])
        return np.array(filtered_saccades)


    def _track_outlier_saccades(self, original_saccades, filtered_saccades, x, y):
        """
        Tracks the saccades that were removed as outliers due to centroid proximity to the boundary.
        Args:
            original_saccades (np.ndarray): The original saccades or microsaccades detected, shape (N, 2).
            filtered_saccades (np.ndarray): The filtered saccades after removing outliers, shape (M, 2).
            x (np.ndarray): x-coordinates of eye position.
            y (np.ndarray): y-coordinates of eye position.
        
        Returns:
            dict: Outlier saccades with their start and stop indices, times, and corresponding XY coordinates.
        """
        # Ensure original_saccades and filtered_saccades are 2D arrays with shape (N, 2)
        if original_saccades.ndim != 2 or original_saccades.shape[1] != 2:
            raise ValueError("original_saccades must be a 2D array of shape (N, 2)")

        if filtered_saccades.ndim != 2 or filtered_saccades.shape[1] != 2:
            raise ValueError("filtered_saccades must be a 2D array of shape (M, 2)")

        # Find the removed saccades by checking which ones are not in the filtered_saccades
        removed_saccades_mask = np.isin(original_saccades, filtered_saccades, invert=True).all(axis=1)
        removed_saccade_indices = original_saccades[removed_saccades_mask]

        # Initialize the outlier info dictionary
        outlier_info = {
            'start_stop_indices': [],  # list of start and stop indices for each removed saccade
            'times': [],  # list of start and stop times for each removed saccade
            'XY': []  # list of (x, y) coordinates for each removed saccade
        }

        # Process each removed saccade
        for saccade in removed_saccade_indices:
            start, end = saccade
            outlier_info['start_stop_indices'].append([start, end])
            outlier_info['times'].append([start * self.samprate, end * self.samprate])

            # Extract the corresponding XY points for the outlier saccade
            outlier_x = x[start:end+1]
            outlier_y = y[start:end+1]
            outlier_info['XY'].append([outlier_x, outlier_y])

        return outlier_info

