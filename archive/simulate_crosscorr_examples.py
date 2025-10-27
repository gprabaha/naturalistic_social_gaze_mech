import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False

import curate_data

def _initialize_params():
    params = {
        'is_cluster': False,
        'prabaha_local': True,
        'is_grace': False,
        'use_parallel': False
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    return params

def generate_binary_vector(length, probability, periodicity=False):
    """Generate a binary vector with a given probability of 1s."""
    vec = np.random.rand(length) < probability
    if periodicity:
        # Introduce some periodicity by adding a sine wave pattern
        sine_wave = (np.sin(np.linspace(0, 10 * np.pi, length)) > 0).astype(int)
        vec = np.logical_or(vec, sine_wave).astype(int)
    return vec

def fft_crosscorr(x, y):
    """Compute normalized cross-correlation using FFT."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x = x - x_mean
    y = y - y_mean
    corr = fftconvolve(x, y[::-1], mode='full')
    norm_factor = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return corr[n-1:] / norm_factor  # Only positive lags

def compute_mean_crosscorr(v1, v2, num_shuffles=100):
    """Compute mean cross-correlation over multiple shuffles."""
    shuffled_corrs = []
    for _ in range(num_shuffles):
        np.random.shuffle(v1)
        np.random.shuffle(v2)
        shuffled_corrs.append(fft_crosscorr(v1, v2))
    return np.mean(shuffled_corrs, axis=0)

def plot_crosscorr(params):
    """Generate example traces and plot their cross-correlations."""
    length = 1000
    num_shuffles = 100
    print("Generating binary vectors")
    # Generate binary vectors
    v1_90 = generate_binary_vector(length, 0.9)
    v2_90 = generate_binary_vector(length, 0.9)
    v1_10 = generate_binary_vector(length, 0.1)
    v2_10 = generate_binary_vector(length, 0.1)
    v1_40_periodic = generate_binary_vector(length, 0.4, periodicity=True)
    v2_40_periodic = generate_binary_vector(length, 0.4, periodicity=True)

    print("Computing cross-correlations")
    # Compute mean cross-correlation for shuffled versions
    mean_corr_90 = compute_mean_crosscorr(v1_90, v2_90, num_shuffles)
    mean_corr_10 = compute_mean_crosscorr(v1_10, v2_10, num_shuffles)

    # Compute cross-correlation for periodic vectors
    corr_40_periodic = fft_crosscorr(v1_40_periodic, v2_40_periodic)

    # Define time lags
    taus = np.arange(len(mean_corr_90))
    print("Plotting cross-correlations")
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(taus, mean_corr_90, color='black', label="Mean 90% 1s")
    plt.plot(taus, mean_corr_10, color='dimgray', label="Mean 10% 1s")
    plt.plot(taus, corr_40_periodic, color='red', label="40% 1s with periodicity")
    plt.xlabel("Tau (lag)")
    plt.ylabel("Cross-correlation")
    plt.title("Cross-Correlation of Example Binary Vectors")
    plt.legend()
    plt.grid(True)

    # Save plot
    root_data_dir = params.get("root_data_dir")  # Replace with actual path from params
    plot_dir = os.path.join(root_data_dir, "plots", "cross_correlation_simulation")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "simulated_crosscorr.pdf")
    plt.savefig(plot_path, format="pdf")
    plt.close()

    print(f"Plot saved to {plot_path}")

# Run the function
params = _initialize_params()
plot_crosscorr(params)
