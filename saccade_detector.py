
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm


def detect_saccades_and_microsaccades_in_position_array(positions, session_name, samprate=1/1000):
    sacc_params = _get_saccade_parameters(session_name, samprate)
    if positions.shape[0] > int(30 / (sacc_params['samprate'] * 1000)):
        print("\nPreprocessing positions data for saccade detection")
        x, y = _preprocess_data(positions, sacc_params)
        saccades_start_stop_inds, microsaccades_start_stop_inds = _detect_saccades_mayo_2023(x, y, sacc_params)
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
    saccade_params = {
        'session_name': session_name,
        'samprate': samprate,
        'num_cpus': num_cpus,
        'fltord': fltord,
        'lowpasfrq': lowpasfrq,
        'nyqfrq': nyqfrq,
        'flt': flt,
        'buffer': buffer
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



def _detect_saccades_mayo_2023(x, y, sacc_params):
    print("Detecting saccades and microsaccades using Willett and Mayo 2023 method")
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
