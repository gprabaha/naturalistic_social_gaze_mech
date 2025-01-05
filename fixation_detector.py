
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def detect_fixation_in_position_array(positions, session_name, samprate=1/1000):

    fix_params = _get_fixation_parameters(session_name)
    if len(positions[0]) > int(30 / (fix_params['samprate'] * 1000)):
        print("Preprocessing positions data for fixation detection")
        x, y = _preprocess_data(positions, fix_params)




    else:
        print("!! Data too short for fixation detectionprocessing !!")
            return {
                'fixationindices': [],
                'XY': np.array([positions[0], positions[1]])
            }


def _get_fixation_parameters(session_name=None, samprate=1/1000, params=None, num_cpus=1):
    # Initialize parameters
    use_parallel = params.get('use_parallel', False) if params else False
    variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
    fltord = 60
    lowpasfrq = 30
    nyqfrq = 1000 / 2  # Nyquist frequency
    flt = signal.firwin2(fltord, 
                         [0, lowpasfrq / nyqfrq, lowpasfrq / nyqfrq, 1], 
                         [1, 1, 0, 0])
    buffer = int(100 / (samprate * 1000))
    # Construct the parameters dictionary
    fixation_params = {
        'session_name': session_name,
        'samprate': samprate,
        'num_cpus': num_cpus,
        'use_parallel': use_parallel,
        'variables': variables,
        'fltord': fltord,
        'lowpasfrq': lowpasfrq,
        'nyqfrq': nyqfrq,
        'flt': flt,
        'buffer': buffer
    }
    return fixation_params


def _preprocess_data(positions, fix_params):
    """Pads, resamples, and filters the eye data.
    Args:
        eyedat (list): List containing x and y coordinates of eye data.
    Returns:
        tuple: Preprocessed x and y coordinates.
    """
    print("Preprocessing x and y data")
    x = np.pad(positions[0], (fix_params['buffer'], fix_params['buffer']), 'reflect')
    y = np.pad(positions[1], (fix_params['buffer'], fix_params['buffer']), 'reflect')
    x = __resample_data(x, fix_params)
    y = __resample_data(y, fix_params)
    x = __apply_filter(x, fix_params)
    y = __apply_filter(y, fix_params)
    x = x[fix_params['buffer']:-fix_params['buffer']]
    y = y[fix_params['buffer']:-fix_params['buffer']]
    return x, y


def __resample_data(data, fix_params):
    """Resamples the data based on the sampling rate.
    Args:
        data (np.ndarray): Array of eye data.
    Returns:
        np.ndarray: Resampled data.
    """
    t_old = np.linspace(0, len(data) - 1, len(data))
    resample_factor = fix_params['samprate'] * 1000
    if resample_factor > 1:
        print(f"Resample factor is too large: {resample_factor}")
        raise ValueError("Resample factor is too large, leading to excessive memory usage.")
    t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
    f = interp1d(t_old, data, kind='linear')
    return f(t_new)


def __apply_filter(data, fix_params):
    """Applies a low-pass filter to the data.
    Args:
        data (np.ndarray): Array of data to be filtered.
    Returns:
        np.ndarray: Filtered data.
    """
    return signal.filtfilt(fix_params['flt'], 1, data)