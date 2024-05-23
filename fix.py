#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:34:44 2024

@author: pg496
"""
"""
Script for fixation detection
"""

import numpy as np
import util  # Import utility functions here
from tqdm import tqdm
import pandas as pd

import pdb  # Import the Python debugger if needed


def is_fixation(pos, time, session_name, t1=None, t2=None, minDur=None, maxDur=None, sampling_rate=None):
    """
    Determine fixations based on position and time data.
    Args:
    pos: Position data (x, y).
    time: Time data.
    t1: Spatial parameter t1.
    t2: Spatial parameter t2.
    minDur: Minimum fixation duration.
    sampling_rate: Sampling rate.
    Returns:
    Binary vector indicating fixations (1) and non-fixations (0).
    """
    # Combine position and time into a single data matrix
    data = np.column_stack((pos, time))
    # Calculate sampling rate if not provided
    if sampling_rate is None:
        sampling_rate = 1 / (time[1,:] - time[0,:])
    # Set default values for parameters if not provided
    if minDur is None:
        minDur = 0.05
    if maxDur is None:
        maxDur = 2
    if t2 is None:
        t2 = 15
    if t1 is None:
        t1 = 30
    # Initialize fix_vector
    fix_vector = np.zeros(data.shape[0])
    '''
    Implement a proper outlier detection code here. The curve fit should
    account for a situation where t_n is (x_a, y_a) and t_m is (x_a, y_b), so
    the same x will have 2 y values, which can get complicated for polyfit
    '''
    fix_list, fix_t_inds = fixation_detection(data, t1, t2, minDur, maxDur, session_name)
    for t_range in fix_t_inds:
        fix_vector[t_range[0]:t_range[1] + 1] = 1
    return fix_list, fix_vector


def fixation_detection(data, t1, t2, minDur, maxDur, session_name):
    """
    Detect fixations based on position and time data.
    Args:
    data: Combined position and time data.
    t1: Spatial parameter t1.
    t2: Spatial parameter t2.
    minDur: Minimum fixation duration.
    Returns:
    List of fixation time ranges.
    """
    n = len(data)
    if n == 0:
        return []  # Return empty list if data is empty
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    n = len(t)
    fixations = get_t1_filtered_fixations(n, x, y, t, t1, session_name)
    number_fixations = fixations[-1, 3]
    fixation_list = []
    for i in tqdm(range(1, int(number_fixations) + 1), desc=f"{session_name}: n fixations t2 filtered"):
        fixation_list.append(filter_fixations_t2(i, fixations, t2))
    # Duration thresholding
    fixation_list = min_duration(fixation_list, minDur)
    fixation_list = max_duration(fixation_list, maxDur)
    # Convert fixation list to time ranges
    fix_ranges = []
    for fix in fixation_list:
        s_ind = np.where(data[:, 2] == fix[4])[0][0]
        e_ind = np.where(data[:, 2] == fix[5])[0][-1]
        fix_ranges.append([s_ind, e_ind])
    col_names = ['fix_x', 'fix_y', 'threshold_1', 'threshold_2', 'start_time', 'end_time', 'duration']
    return pd.DataFrame(fixation_list, columns=col_names), fix_ranges


def get_t1_filtered_fixations(n, x, y, t, t1, session_name):
    """
    Filter fixations based on spatial parameter t1.
    Args:
    n: Length of data.
    x: X-coordinate data.
    y: Y-coordinate data.
    t: Time data.
    t1: Spatial parameter t1.
    Returns:
    Array of fixations after filtering.
    """
    fixations = np.zeros((n, 4))
    fixid = 0
    fixpointer = 0
    for i in tqdm(range(n), desc='{}: n data points t1 filtered'.format(session_name)):
        if not np.any(x[fixpointer:i+1]) or not np.any(y[fixpointer:i+1]):
            fixations = update_fixations(i, x, y, t, fixations, fixid)
        else:
            mx = np.mean(x[fixpointer:i+1])
            my = np.mean(y[fixpointer:i+1])
            d = util.distance2p(mx, my, x[i], y[i])
            if d > t1:
                fixid += 1
                fixpointer = i
            fixations = update_fixations(i, x, y, t, fixations, fixid)
    return fixations


def update_fixations(i, x, y, t, fixations, fixid):
    """
    Update fixations array with new fixation data.
    Args:
    i: Index.
    x: X-coordinate data.
    y: Y-coordinate data.
    t: Time data.
    fixations: Array of fixations.
    fixid: ID of the fixation.
    Returns:
    Updated fixations array.
    """
    fixations[i, 0] = x[i]
    fixations[i, 1] = y[i]
    fixations[i, 2] = t[i]
    fixations[i, 3] = fixid
    return fixations


def filter_fixations_t2(fixation_id, fixations, t2):
    """
    Cluster fixations based on spatial criteria and apply t2 threshold.
    Args:
    fixation_id: ID of the fixation.
    fixations: Array containing fixations.
    t2: Spatial parameter t2.
    Returns:
    Fixation information after applying t2 threshold.
    """
    fixations_id = fixations[fixations[:, 3] == fixation_id]
    number_t1 = len(fixations_id)
    # Clustering according to criterion t2
    fixx, fixy = np.nanmean(fixations_id[:, :2], axis=0)
    for i in range(number_t1):
        d = util.distance2p(fixx, fixy, fixations_id[i, 0], fixations_id[i, 1])
        if d > t2:
            fixations_id[i, 3] = 0
    # Initialize lists
    fixations_list_t2 = np.empty((0, 4))
    list_out_points = np.empty((0, 4))
    for i in range(number_t1):
        if fixations_id[i, 3] > 0:
            fixations_list_t2 = np.vstack((fixations_list_t2, fixations_id[i, :]))
        else:
            list_out_points = np.vstack((list_out_points, fixations_id[i, :]))
    # Compute number of t2 fixations
    number_t2 = fixations_list_t2.shape[0]
    if not np.any(fixations_list_t2[:, :2]):
        start_time, end_time, duration = 0, 0, 0
    else:
        fixx, fixy = np.nanmean(fixations_list_t2[:, :2], axis=0)
        start_time = fixations_list_t2[0, 2]
        end_time = fixations_list_t2[-1, 2]
        duration = end_time - start_time
    return fixx, fixy, number_t1, number_t2, start_time, end_time, duration


def min_duration(fixation_list, minDur):
    """
    Apply duration threshold to fixation list.
    Args:
    fixation_list: List of fixations.
    minDur: Minimum fixation duration.
    Returns:
    Fixation list after applying duration threshold.
    """
    return [fix for fix in fixation_list if fix[6] >= minDur]

def max_duration(fixation_list, maxDur):
    return [fix for fix in fixation_list if fix[6] <= maxDur]