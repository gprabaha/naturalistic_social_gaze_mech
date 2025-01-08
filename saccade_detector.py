
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

import defaults


def detect_saccades_and_microsaccades_in_position_array(positions, session_name, samprate=1/1000):
    sacc_params = _get_saccade_parameters(session_name, samprate)
    if positions.shape[0] > int(30 / (sacc_params['samprate'] * 1000)):
        print("\nPreprocessing positions data for saccade detection")
        x, y = _preprocess_data(positions, sacc_params)
        saccades_start_stop_inds, microsaccades_start_stop_inds = _detect_saccades_hubel_2000(x, y, sacc_params)
        return saccades_start_stop_inds, microsaccades_start_stop_inds
    else:
        print("\n!! Data too short for saccade detection processing !!\n")
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int) 



def _get_saccade_parameters(session_name=None, samprate=1/1000, num_cpus=1):
    fltord = 60
    lowpasfrq = 30
    nyqfrq = 1000 / 2  # Nyquist frequency
    flt = signal.firwin2(fltord, 
                         [0, lowpasfrq / nyqfrq, lowpasfrq / nyqfrq, 1], 
                         [1, 1, 0, 0])
    buffer = int(100 / (samprate * 1000))
    monitor_info = defaults.fetch_monitor_info()
    saccade_params = {
        'session_name': session_name,
        'samprate': samprate,
        'num_cpus': num_cpus,
        'fltord': fltord,
        'lowpasfrq': lowpasfrq,
        'nyqfrq': nyqfrq,
        'flt': flt,
        'buffer': buffer,
        'monitor_info': monitor_info
    }
    return saccade_params



def _preprocess_data(positions, sacc_params):
    print("Preprocessing x and y data")
    x = np.pad(positions[:, 0], (sacc_params['buffer'], sacc_params['buffer']), 'reflect')
    y = np.pad(positions[:, 1], (sacc_params['buffer'], sacc_params['buffer']), 'reflect')
    x = __resample_data(x, sacc_params)
    y = __resample_data(y, sacc_params)
    x = __apply_filter(x, sacc_params)
    y = __apply_filter(y, sacc_params)
    x = x[sacc_params['buffer']:-sacc_params['buffer']]
    y = y[sacc_params['buffer']:-sacc_params['buffer']]
    return x, y

def __resample_data(data, sacc_params):
    t_old = np.linspace(0, len(data) - 1, len(data))
    resample_factor = sacc_params['samprate'] * 1000
    if resample_factor > 1:
        print(f"Resample factor is too large: {resample_factor}")
        raise ValueError("Resample factor is too large, leading to excessive memory usage.")
    t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
    f = interp1d(t_old, data, kind='linear')
    return f(t_new)

def __apply_filter(data, sacc_params):
    return signal.filtfilt(sacc_params['flt'], 1, data)



def _detect_saccades_hubel_2000(x, y, sacc_params):
    """Detects saccades and microsaccades using the method inspired by Martinez-Conde, Macknik, and Hubel (2000).
    The algorithm applies a velocity threshold and a direction change criterion to determine eye movement events. 
    This method is informed by the importance of identifying fine-grained eye movements such as microsaccades, 
    which play a crucial role in visual perception and attention, as highlighted in the 2013 Nature Reviews Neuroscience article.
    Args:
        data (tuple): Tuple containing x and y coordinates of eye position in pixels.
    Returns:
        dict: Indices and time points of saccades and microsaccades with start and end indices.
    """
    print("Using Hubel et al. (2000) method for saccade detection")
    # Convert x, y from pixels to degrees
    x_deg, y_deg = __pixels_to_degrees(x, y, sacc_params)
    dx = np.diff(x_deg)
    dy = np.diff(y_deg)
    velocity = np.sqrt(dx**2 + dy**2) / sacc_params['samprate']  # Convert to degrees per second
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
    return np.array(saccades), np.array(microsaccades)


def __pixels_to_degrees(x, y, sacc_params):
    """Converts pixel coordinates to degrees of visual angle, accounting for monitor size and viewing distance.
    Args:
        x (np.ndarray): Array of x-coordinates in pixels.
        y (np.ndarray): Array of y-coordinates in pixels.
    Returns:
        tuple: Two numpy arrays representing x and y coordinates in degrees of visual angle.
    """
    # Monitor specifications
    monitor_diagonal_inches = sacc_params['monitor_info']['diagonal']  # inches
    monitor_distance_cm = sacc_params['monitor_info']['distance']  # cm
    vertical_res = sacc_params['monitor_info']['vertical_resolution']  # pixels
    horizontal_res = sacc_params['monitor_info']['horizontal_resolution']  # pixels
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


def _detect_saccades_mayo_2023(x, y, sacc_params):
    '''
    I AM NOT SURE THIS IS THE RIGHT WAY OF DETECTING SACCADES. THE ANGULAR VELOCITY THRESHOLDS COULD BE DVA
    BASED INSTEAD OF THE WAY WE ARE CALCULATING IT RIGHT NOW. NEED TO MAKE SURE THAT THE SACCADE DETECTOINS 
    IS CORRECT, AND THEN HOW TO DEAL WITH CLASHES WITH FIXATION WINDOWS
    '''

    print("Detecting saccades and microsaccades using Willett and Mayo 2023 method")

    '''
    x_deg, y_deg = self.pixels_to_degrees(x, y)
    dx = np.diff(x_deg)
    dy = np.diff(y_deg)
    velocity = np.sqrt(dx**2 + dy**2) / self.samprate
    '''

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    radial_velocity = np.diff(r) * sacc_params['samprate']
    radial_velocity = np.append(radial_velocity, radial_velocity[-1])
    smoothed_velocity = uniform_filter1d(np.abs(radial_velocity), size=3)
    # Remove eye movement noise and calculate thresholds
    eye_movement_removed = smoothed_velocity[smoothed_velocity < 6]
    mean_velocity = np.mean(eye_movement_removed)
    std_velocity = np.std(eye_movement_removed)
    onset_threshold = mean_velocity + std_velocity
    # Find saccade peaks
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
    # Classify saccades and microsaccades
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
    print(f"Detected {len(saccades)} saccades and {len(microsaccades)} microsaccades.")
    return np.array(saccades), np.array(microsaccades)
