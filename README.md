# Naturalistic Social Gaze Mechanisms

This repository investigates patterns of behavior observed during naturalistic gaze exchange between two primates. Simultaneous timeseries of gaze locations recorded from two individuals are analyzed to identify stereotypical fixation and saccade behaviors. By modeling the transition probabilities of socially relevant behaviors, this project aims to understand inter-agent interactions in naturalistic settings.

## Project Structure

The code in this repository is structured into different modules, each responsible for various stages of data preprocessing, analysis, and plotting. Here’s a brief overview of the key files:

### `analyze_gaze_signals.py`
This is the main entry point to the analysis pipeline. Running this script will process the gaze data and compute various metrics associated with fixation and saccade events between the two primates.

**Usage:**
python analyze_gaze_signals.py

### `data_manager.py`
Contains the `DataManager` class, which handles the coordination of data loading, processing, and analysis. The `DataManager` calls functions from different modules (e.g., `curate_data.py`, `fix_and_saccades.py`, and `analyze_data.py`) to perform the required computations on the gaze data.

Key methods in `DataManager`:
- **`get_data()`**: Loads or computes gaze data.
- **`prune_data()`**: Cleans the data by removing NaN values and irrelevant time points.
- **`analyze_behavior()`**: Detects fixations and saccades, creates binary behavioral timeseries, and computes autocorrelations.
- **`plot_behavior()`**: (Currently under revision) Will handle the plotting of behavioral patterns.

### Core Modules:
- **`curate_data.py`**: Responsible for preparing and cleaning the data. This includes functions to prune missing values, synchronize timestamps, and manage file paths.
- **`fix_and_saccades.py`**: Contains the logic for detecting fixation and saccade events from gaze position data.
- **`analyze_data.py`**: Computes various analyses, including creating binary behavioral timeseries and calculating autocorrelations of behavioral events.
- **`load_data.py`**: Contains functions to load precomputed data, including gaze timeseries, fixations, saccades, and autocorrelations.
- **`plotter.py`**: (Work in progress) Will handle the visualization of results, including plots of gaze patterns, fixations, saccades, and their transitions.

### Utilities:
- **`util.py`**: A collection of utility functions used across the repository for tasks like logging, file path management, and progress tracking.
- **`defaults.py`**: Contains configuration defaults, such as monitor specifications and default parameters for fixation/saccade detection.

## Setup and Installation

### Prerequisites:
- Python 3.8 or higher
- Required Python packages are listed in `requirements.txt`. Install them with:
    pip install -r requirements.txt

### Running the Analysis:
To run the full analysis pipeline, use the following command:
python analyze_gaze_signals.py

This script will load the gaze data, perform fixation and saccade detection, compute behavioral metrics, and (eventually) generate visualizations.

## Repository Workflow:

1. **Data Loading and Cleaning**: The gaze data is first loaded via the `get_data()` method in `DataManager`, which ensures that missing or incorrect data is handled appropriately.
2. **Fixation and Saccade Detection**: Using `fix_and_saccades.py`, the gaze data is analyzed to detect periods of fixation and saccade behavior, which are critical for understanding gaze dynamics.
3. **Binary Timeseries Creation**: The detected behaviors are translated into binary timeseries (fixation and saccade vectors) in `analyze_data.py` for subsequent analysis.
4. **Autocorrelation Analysis**: The autocorrelations of the binary timeseries are computed to investigate behavior patterns over time.
5. **Future Visualization**: Although under development, the `plot_behavior()` method in `DataManager` will generate visualizations of the analyzed gaze behavior.

## Contributing
Contributions are welcome! If you have any suggestions or find any issues, feel free to open a pull request or issue on the [GitHub repository](https://github.com/gprabaha/naturalistic_social_gaze_mech).
