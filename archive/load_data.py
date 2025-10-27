#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:56:31 2024

@author: pg496
"""

import os
import pickle
import logging
import scipy.io
import numpy as np
import pandas as pd


# Set up a logger for this module
logger = logging.getLogger(__name__)


def load_mat_from_path(path):
    return scipy.io.loadmat(path)


def load_recording_days(data_file_path="ephys_days_and_monkeys.pkl"):
    """Load the recording days list from a pickle file, if it exists."""
    try:
        with open(data_file_path, 'rb') as f:
            return pd.DataFrame(pickle.load(f))
    except FileNotFoundError:
        return []


def get_gaze_data_df(gaze_data_file_path, missing_data_file_path):
    """
    Loads the gaze data dictionary and missing data paths from saved pickle files.
    Parameters:
    - gaze_data_file_path (str): Path to the saved gaze data dictionary file.
    - missing_data_file_path (str): Path to the saved missing data paths file.
    Returns:
    - gaze_data_dict (dict): The loaded gaze data dictionary.
    - missing_data_paths (list): The loaded list of missing data paths.
    """
    try:
        # Load gaze data dictionary
        with open(gaze_data_file_path, 'rb') as f:
            gaze_data_df = pickle.load(f)
        # Load missing data paths
        with open(missing_data_file_path, 'rb') as f:
            missing_data_paths = pickle.load(f)
        return gaze_data_df, missing_data_paths
    except Exception as e:
        logger.error(f"Failed to load gaze data or missing data paths: {e}")
        raise


def get_spike_times_df(file_path):
    try:
        with open(file_path, 'rb') as f:
            gaze_data_df = pickle.load(f)
        return gaze_data_df
    except Exception as e:
        logger.error(f"Failed to load gaze data: {e}")
        raise
    

def get_data_df(data_file_path):
    """
    Loads the gaze data dictionary from a saved pickle file.
    Parameters:
    - data_file_path (str): Path to the saved gaze data dictionary file.
    Returns:
    - data_df (pd.DataFrame or dict): The loaded gaze data dictionary.
    """
    try:
        with open(data_file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"pickle.load failed: {e}. Trying pd.read_pickle instead.")
        try:
            return pd.read_pickle(data_file_path)
        except Exception as e:
            logger.error(f"Both pickle.load and pd.read_pickle failed: {e}")
            raise


def load_fixation_and_saccade_dfs(fixation_file_path, saccade_file_path):
    """
    Loads the fixation and saccade dataframes from the processed data directory.
    Parameters:
    - params (dict): A dictionary containing configuration parameters, including the processed data directory path.
    Returns:
    - fixation_df (dataframe): The fixation data loaded from the saved pickle file.
    - saccade_df (dataframe): The saccade data loaded from the saved pickle file.
    """
    try:
        # Load fixation dictionary
        with open(fixation_file_path, 'rb') as f:
            fixation_df = pickle.load(f)
        logger.info(f"Successfully loaded fixation data from {fixation_file_path}")
        # Load saccade dictionary
        with open(saccade_file_path, 'rb') as f:
            saccade_df = pickle.load(f)
        logger.info(f"Successfully loaded saccade data from {saccade_file_path}")
        return fixation_df, saccade_df
    except Exception as e:
        logger.error(f"Failed to load fixation or saccade data: {e}")
        raise


# Function to load binary behavior timeseries DataFrame
def load_binary_timeseries_df(file_path):
    """
    Loads the binary behavior timeseries DataFrame from a pickle file.
    Parameters:
    file_path (str): The file path to the pickle file.
    Returns:
    pd.DataFrame: The binary behavior timeseries DataFrame.
    """
    with open(file_path, 'rb') as f:
        binary_timeseries_df = pickle.load(f)
    logger.info(f"Successfully loaded binary timeseries df from {file_path}")
    return binary_timeseries_df


# Function to load binary timeseries autocorrelation DataFrame
def load_binary_crosscorr_df(file_path):
    """
    Loads the binary timeseries autocorrelation DataFrame from a pickle file.
    Parameters:
    file_path (str): The file path to the pickle file.
    Returns:
    pd.DataFrame: The binary timeseries autocorrelation DataFrame.
    """
    with open(file_path, 'rb') as f:
        binary_autocorr_df = pickle.load(f)
    logger.info(f"Successfully loaded cross-correlation df from {file_path}")
    return binary_autocorr_df


def load_neural_timeseries_df(file_path):
    """
    Load scaled autocorrelations from a pickle file.
    Parameters:
    - file_path (str): The file path to the pickle file.
    Returns:
    - dict: Loaded data from 'scaled_autocorrelations.pkl' if file exists, else None.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, 'rb') as f:
        scaled_autocorrelations = pickle.load(f)
    return scaled_autocorrelations

