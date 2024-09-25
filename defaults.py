#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:36:16 2024

@author: prabaha
"""

from scipy.signal.windows import gaussian
from numpy import convolve

def fetch_monitor_info():
    return {'diagonal': 17,
            'distance': 50,
            'vertical_resolution': 1024,
            'horizontal_resolution': 1280}

def fetch_default_saccade_pars():
    window = gaussian(21, 5, True)
    smooth_func = lambda x: convolve(x, window, mode='same')
    return {'vel_thresh': [50, 1000],
            'min_samples': 50,
            'smooth_func': smooth_func}
