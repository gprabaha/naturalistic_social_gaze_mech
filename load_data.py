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


# Set up a logger for this module
logger = logging.getLogger(__name__)


def load_mat_from_path(path):
    return scipy.io.loadmat(path)


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


def get_nan_removed_gaze_data_df(nan_removed_gaze_data_file_path):
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
        with open(nan_removed_gaze_data_file_path, 'rb') as f:
            gaze_data_df = pickle.load(f)
        return gaze_data_df
    except Exception as e:
        logger.error(f"Failed to load gaze data: {e}")
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






