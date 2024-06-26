#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:44:40 2024

@author: pg496
"""

import numpy as np
import util
from scipy import signal
from scipy.interpolate import interp1d

class EyeMVMSaccadeDetector:
    def __init__(self, vel_thresh, min_samples, sampling_rate):
        self.vel_thresh = vel_thresh
        self.min_samples = min_samples
        self.sampling_rate = sampling_rate
        self.fltord = 60
        self.lowpasfrq = 30
        self.nyqfrq = (1 / sampling_rate) / 2
        self.flt = signal.firwin2(self.fltord, [0, self.lowpasfrq / self.nyqfrq, self.lowpasfrq / self.nyqfrq, 1], [1, 1, 0, 0])
        self.buffer = int(100 / (self.sampling_rate * 1000))

    def extract_saccades_for_session(self, session_data):
        positions, info = session_data
        sampling_rate = info['sampling_rate']
        n_samples = positions.shape[0]
        time_vec = util.create_timevec(n_samples, sampling_rate)
        positions = self.preprocess_data(positions)
        session_saccades = self.extract_saccades(positions, time_vec, info)
        return session_saccades

    def preprocess_data(self, eyedat):
        x = np.pad(eyedat[:, 0], (self.buffer, self.buffer), 'reflect')
        y = np.pad(eyedat[:, 1], (self.buffer, self.buffer), 'reflect')
        x = self.resample_data(x)
        y = self.resample_data(y)
        x = self.apply_filter(x)
        y = self.apply_filter(y)
        x = x[100:-100]
        y = y[100:-100]
        return np.column_stack((x, y))

    def resample_data(self, data):
        t_old = np.linspace(0, len(data) - 1, len(data))
        resample_factor = self.sampling_rate * 1000
        if resample_factor > 1:
            print(f"Resample factor is too large: {resample_factor}")
            raise ValueError("Resample factor is too large, leading to excessive memory usage.")
        t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
        f = interp1d(t_old, data, kind='linear')
        return f(t_new)

    def apply_filter(self, data):
        return signal.filtfilt(self.flt, 1, data)

    def extract_saccades(self, positions, time_vec, info):
        session_saccades = []
        category = info['category']
        session_name = info['session_name']
        n_runs = info['num_runs']
        for run in range(n_runs):
            run_start = info['startS'][run]
            run_stop = info['stopS'][run]
            run_time = (time_vec > run_start) & (time_vec <= run_stop)
            run_positions = positions[run_time, :]
            run_x = util.px2deg(run_positions[:, 0].T)
            run_y = util.px2deg(run_positions[:, 1].T)
            saccade_start_stops = self.find_saccades(run_x, run_y, info['sampling_rate'])
            for start, stop in saccade_start_stops:
                saccade = run_positions[start:stop + 1, :]
                start_time = time_vec[start]
                end_time = time_vec[stop]
                duration = end_time - start_time
                start_roi = self.determine_roi_of_coord(run_positions[start, :2], info['roi_bb_corners'])
                end_roi = self.determine_roi_of_coord(run_positions[stop, :2], info['roi_bb_corners'])
                block = self.determine_block(start_time, end_time, info['startS'], info['stopS'])
                session_saccades.append([start_time, end_time, duration, saccade, start_roi, end_roi, session_name, category, run, block])
        return session_saccades

    def find_saccades(self, x, y, sr):
        assert x.shape == y.shape
        start_stops = []
        x0 = self.apply_filter(x)
        y0 = self.apply_filter(y)
        vx = np.gradient(x0) / sr
        vy = np.gradient(y0) / sr
        vel_norm = np.sqrt(vx ** 2 + vy ** 2)
        above_thresh = (vel_norm >= self.vel_thresh[0]) & (vel_norm <= self.vel_thresh[1])
        start_stops = util.find_islands(above_thresh, self.min_samples)
        return start_stops

    def determine_roi_of_coord(self, position, bbox_corners):
        bounding_boxes = ['eye_bbox', 'face_bbox', 'left_obj_bbox', 'right_obj_bbox']
        inside_roi = [util.is_inside_roi(position, bbox_corners[key]) for key in bounding_boxes]
        if any(inside_roi):
            if inside_roi[0] and inside_roi[1]:
                return bounding_boxes[0]
            return bounding_boxes[inside_roi.index(True)]
        return 'out_of_roi'

    def determine_block(self, start_time, end_time, startS, stopS):
        if start_time < startS[0] or end_time > stopS[-1]:
            return 'discard'
        for i, (run_start, run_stop) in enumerate(zip(startS, stopS), start=1):
            if start_time >= run_start and end_time <= run_stop:
                return 'mon_down'
            elif i < len(startS) and end_time <= startS[i]:
                return 'mon_up'
        return 'discard'
