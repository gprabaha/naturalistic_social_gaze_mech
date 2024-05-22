#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:23:31 2024

@author: pg496
"""


def get_root_data_dir(params):
    """
    Returns the root data directory based on whether it's running on a cluster or not.
    Parameters:
    - params (dict): Dictionary containing parameters.
    Returns:
    - root_data_dir (str): Root data directory path.
    """
    is_cluster = params['is_cluster']
    return "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/" if is_cluster \
        else "/Volumes/Stash/changlab/social_gaze"


