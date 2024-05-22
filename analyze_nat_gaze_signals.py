#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:19:31 2024

@author: pg496
"""

import numpy as np
import os
import scipy

import util
import filter_behav
import filter_ephys
import plotter

params = {}
params.update({
    'is_cluster': True,
    'use_parallel': False})

root_data_dir = util.get_root_data_dir(params)
