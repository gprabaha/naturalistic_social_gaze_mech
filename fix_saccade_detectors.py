#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:54:12 2024

@author: pg496
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import util
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count


class EyeMVMFixationDetector:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
        self.fltord = 60
        self.lowpasfrq = 30
        self.nyqfrq = (1 / sampling_rate) / 2
        self.flt = signal.firwin2(self.fltord, [0, self.lowpasfrq / self.nyqfrq, self.lowpasfrq / self.nyqfrq, 1], [1, 1, 0, 0])
        self.buffer = int(100 / (self.sampling_rate * 1000))

    def detect_fixations(self, positions, time_vec, session_name):
        positions = self.preprocess_data(positions)
        fix_timepos_df, fix_vec_entire_session = self.is_fixation(positions, time_vec, session_name)
        fixationtimes = fix_timepos_df[['start_time', 'end_time']].to_numpy().T
        fixations = fix_timepos_df[['fix_x', 'fix_y']].to_numpy().T
        return fixationtimes, fixations

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

    def is_fixation(self, pos, time, session_name, t1=None, t2=None, minDur=None, maxDur=None):
        data = np.column_stack((pos, time))
        if minDur is None:
            minDur = 0.05
        if maxDur is None:
            maxDur = 2
        if t2 is None:
            t2 = 15
        if t1 is None:
            t1 = 30
        fix_vector = np.zeros(data.shape[0])
        fix_list_df, fix_t_inds = self.fixation_detection(data, t1, t2, minDur, maxDur, session_name)
        for t_range in fix_t_inds:
            fix_vector[t_range[0]:t_range[1] + 1] = 1
        return fix_list_df, fix_vector

    def fixation_detection(self, data, t1, t2, minDur, maxDur, session_name):
        n = len(data)
        if n == 0:
            return []
        x = data[:, 0]
        y = data[:, 1]
        t = data[:, 2]
        fixations = self.get_t1_filtered_fixations(n, x, y, t, t1, session_name)
        number_fixations = fixations[-1, 3]
        fixation_list = []
        for i in tqdm(range(1, int(number_fixations) + 1), desc=f"{session_name}: n fixations t2 filtered"):
            fixation_list.append(self.filter_fixations_t2(i, fixations, t2))
        fixation_list = self.min_duration(fixation_list, minDur)
        fixation_list = self.max_duration(fixation_list, maxDur)
        fix_ranges = []
        for fix in fixation_list:
            s_ind = np.where(data[:, 2] == fix[4])[0][0]
            e_ind = np.where(data[:, 2] == fix[5])[0][-1]
            fix_ranges.append([s_ind, e_ind])
        col_names = ['fix_x', 'fix_y', 'threshold_1', 'threshold_2', 'start_time', 'end_time', 'duration']
        return pd.DataFrame(fixation_list, columns=col_names), fix_ranges

    def get_t1_filtered_fixations(self, n, x, y, t, t1, session_name):
        fixations = np.zeros((n, 4))
        fixid = 0
        fixpointer = 0
        for i in tqdm(range(n), desc=f'{session_name}: n data points t1 filtered'):
            if not np.any(x[fixpointer:i + 1]) or not np.any(y[fixpointer:i + 1]):
                fixations = self.update_fixations(i, x, y, t, fixations, fixid)
            else:
                mx = np.mean(x[fixpointer:i + 1])
                my = np.mean(y[fixpointer:i + 1])
                d = util.distance2p(mx, my, x[i], y[i])
                if d > t1:
                    fixid += 1
                    fixpointer = i
                fixations = self.update_fixations(i, x, y, t, fixations, fixid)
        return fixations

    def update_fixations(self, i, x, y, t, fixations, fixid):
        fixations[i, 0] = x[i]
        fixations[i, 1] = y[i]
        fixations[i, 2] = t[i]
        fixations[i, 3] = fixid
        return fixations

    def filter_fixations_t2(self, fixation_id, fixations, t2):
        fixations_id = fixations[fixations[:, 3] == fixation_id]
        number_t1 = len(fixations_id)
        fixx, fixy = np.nanmean(fixations_id[:, :2], axis=0)
        for i in range(number_t1):
            d = util.distance2p(fixx, fixy, fixations_id[i, 0], fixations_id[i, 1])
            if d > t2:
                fixations_id[i, 3] = 0
        fixations_list_t2 = np.empty((0, 4))
        list_out_points = np.empty((0, 4))
        for i in range(number_t1):
            if fixations_id[i, 3] > 0:
                fixations_list_t2 = np.vstack((fixations_list_t2, fixations_id[i, :]))
            else:
                list_out_points = np.vstack((list_out_points, fixations_id[i, :]))
        number_t2 = fixations_list_t2.shape[0]
        if not np.any(fixations_list_t2[:, :2]):
            start_time, end_time, duration = 0, 0, 0
        else:
            fixx, fixy = np.nanmean(fixations_list_t2[:, :2], axis=0)
            start_time = fixations_list_t2[0, 2]
            end_time = fixations_list_t2[-1, 2]
            duration = end_time - start_time
        return fixx, fixy, number_t1, number_t2, start_time, end_time, duration

    def min_duration(self, fixation_list, minDur):
        return [fix for fix in fixation_list if fix[6] >= minDur]

    def max_duration(self, fixation_list, maxDur):
        return [fix for fix in fixation_list if fix[6] <= maxDur]
    


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
        self.num_cpus = cpu_count()

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


class ClusterFixationDetector:
    def __init__(self, samprate=1/1000, use_parallel=False):
        self.samprate = samprate
        self.use_parallel = use_parallel
        self.variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
        self.fltord = 60
        self.lowpasfrq = 30
        self.nyqfrq = 1000 / 2
        self.flt = signal.firwin2(self.fltord, [0, self.lowpasfrq / self.nyqfrq, self.lowpasfrq / self.nyqfrq, 1], [1, 1, 0, 0])
        self.buffer = int(100 / (self.samprate * 1000))
        self.fixationstats = []

    def detect_fixations(self, eyedat):
        if not eyedat:
            raise ValueError("No data file found")

        results = []
        with Pool(processes=self.num_cpus) as pool:
            results = pool.map(self.process_eyedat, eyedat)

        self.fixationstats = results
        return self.fixationstats

    def process_eyedat(self, data):
        if len(data[0]) > int(500 / (self.samprate * 1000)):
            x, y = self.preprocess_data(data)
            vel, accel, angle, dist, rot = self.extract_parameters(x, y)
            points = self.normalize_parameters(dist, vel, accel, rot)

            T, meanvalues, stdvalues = self.global_clustering(points)
            fixationcluster, fixationcluster2 = self.find_fixation_clusters(meanvalues, stdvalues)
            T = self.classify_fixations(T, fixationcluster, fixationcluster2)

            fixationindexes, fixationtimes = self.behavioral_index(T, 1)
            fixationtimes = self.apply_duration_threshold(fixationtimes, 25)

            notfixations = self.local_reclustering(fixationtimes, points)
            fixationindexes = self.remove_not_fixations(fixationindexes, notfixations)
            saccadeindexes, saccadetimes = self.classify_saccades(fixationindexes, points)

            fixationtimes, saccadetimes = self.round_times(fixationtimes, saccadetimes)

            pointfix, pointsac, recalc_meanvalues, recalc_stdvalues = self.calculate_cluster_values(fixationtimes, saccadetimes, data)

            return {
                'fixationtimes': fixationtimes,
                'fixations': self.extract_fixations(fixationtimes, data),
                'saccadetimes': saccadetimes,
                'FixationClusterValues': pointfix,
                'SaccadeClusterValues': pointsac,
                'MeanClusterValues': recalc_meanvalues,
                'STDClusterValues': recalc_stdvalues,
                'XY': np.array([data[0], data[1]]),
                'variables': self.variables
            }
        else:
            return {
                'fixationtimes': [],
                'fixations': [],
                'saccadetimes': [],
                'FixationClusterValues': [],
                'SaccadeClusterValues': [],
                'MeanClusterValues': [],
                'STDClusterValues': [],
                'XY': np.array([data[0], data[1]]),
                'variables': self.variables
            }

    def preprocess_data(self, eyedat):
        x = np.pad(eyedat[0], (self.buffer, self.buffer), 'reflect')
        y = np.pad(eyedat[1], (self.buffer, self.buffer), 'reflect')
        x = self.resample_data(x)
        y = self.resample_data(y)
        x = self.apply_filter(x)
        y = self.apply_filter(y)
        x = x[100:-100]
        y = y[100:-100]
        return x, y

    def resample_data(self, data):
        t_old = np.linspace(0, len(data) - 1, len(data))
        resample_factor = self.samprate * 1000
        if resample_factor > 1:
            print(f"Resample factor is too large: {resample_factor}")
            raise ValueError("Resample factor is too large, leading to excessive memory usage.")
        t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
        f = interp1d(t_old, data, kind='linear')
        return f(t_new)

    def apply_filter(self, data):
        return signal.filtfilt(self.flt, 1, data)

    def extract_parameters(self, x, y):
        velx = np.diff(x)
        vely = np.diff(y)
        vel = np.sqrt(velx ** 2 + vely ** 2)
        accel = np.abs(np.diff(vel))
        angle = np.degrees(np.arctan2(vely, velx))
        vel = vel[:-1]
        rot = np.zeros(len(x) - 2)
        dist = np.zeros(len(x) - 2)
        for a in range(len(x) - 2):
            rot[a] = np.abs(angle[a] - angle[a + 1])
            dist[a] = np.sqrt((x[a] - x[a + 2]) ** 2 + (y[a] - y[a + 2]) ** 2)
        rot[rot > 180] -= 180
        rot = 360 - rot
        return vel, accel, angle, dist, rot

    def normalize_parameters(self, dist, vel, accel, rot):
        points = np.stack([dist, vel, accel, rot], axis=1)
        for ii in range(points.shape[1]):
            thresh = np.mean(points[:, ii]) + 3 * np.std(points[:, ii])
            points[points[:, ii] > thresh, ii] = thresh
            points[:, ii] = points[:, ii] - np.min(points[:, ii])
            points[:, ii] = points[:, ii] / np.max(points[:, ii])
        return points

    def global_clustering(self, points):
        if self.use_parallel:
            with Pool(processes=self.num_cpus) as pool:
                futures = [pool.apply_async(self.cluster_and_silhouette, (points, numclusts)) for numclusts in range(2, 6)]
                sil = np.zeros(5)
                for future in tqdm(futures, desc="Global Clustering Progress"):
                    numclusts, score = future.get()
                    sil[numclusts - 2] = score
        else:
            sil = np.zeros(5)
            for numclusts in tqdm(range(2, 6), desc="Global Clustering Progress"):
                sil[numclusts - 2] = self.cluster_and_silhouette(points, numclusts)[1]
        numclusters = np.argmax(sil) + 2
        T = KMeans(n_clusters=numclusters, n_init=5).fit(points)
        labels = T.labels()
        meanvalues = np.array([np.mean(points[labels == i], axis=0) for i in range(numclusters)])
        stdvalues = np.array([np.std(points[labels == i], axis=0) for i in range(numclusters)])
        return labels, meanvalues, stdvalues

    def cluster_and_silhouette(self, points, numclusts):
        T = KMeans(n_clusters=numclusts, n_init=5).fit(points[::10, 1:4])
        silh = self.inter_vs_intra_dist(points[::10, 1:4], T.labels_)
        return numclusts, np.mean(silh)

    def find_fixation_clusters(self, meanvalues, stdvalues):
        fixationcluster = np.argmin(np.sum(meanvalues[:, 1:3], axis=1))
        fixationcluster2 = np.where(meanvalues[:, 1] < meanvalues[fixationcluster, 1] + 3 * stdvalues[fixationcluster, 1])[0]
        fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
        return fixationcluster, fixationcluster2

    def classify_fixations(self, T, fixationcluster, fixationcluster2):
        T[T == fixationcluster] = 100
        for cluster in fixationcluster2:
            T[T == cluster] = 100
        T[T != 100] = 2
        T[T == 100] = 1
        return T

    def behavioral_index(self, T, label):
        indexes = np.where(T == label)[0]
        return indexes, self.find_behavioral_times(indexes)

    def find_behavioral_times(self, indexes):
        dind = np.diff(indexes)
        gaps = np.where(dind > 1)[0]
        if gaps.size > 0:
            behaveind = np.split(indexes, gaps + 1)
        else:
            behaveind = [indexes]
        behaviortime = np.zeros((2, len(behaveind)), dtype=int)
        for i, ind in enumerate(behaveind):
            behaviortime[:, i] = [ind[0], ind[-1]]
        return behaviortime

    def apply_duration_threshold(self, times, threshold):
        return times[:, np.diff(times, axis=0)[0] >= threshold]

    def local_reclustering(self, fixationtimes, points):
        notfixations = []
        if self.use_parallel:
            with Pool(processes=self.num_cpus) as pool:
                futures = [pool.apply_async(self.process_local_reclustering, (fix, points)) for fix in fixationtimes.T]
                for future in tqdm(futures, desc="Local Clustering Progress"):
                    notfixations.extend(future.get())
        else:
            for fix in tqdm(fixationtimes.T, desc="Local Clustering Progress"):
                notfixations.extend(self.process_local_reclustering(fix, points))
    
        return np.array(notfixations)

    def process_local_reclustering(self, fix, points):
        altind = np.arange(fix[0] - 50, fix[1] + 50)
        altind = altind[(altind >= 0) & (altind < len(points))]
        POINTS = points[altind]
        sil = np.zeros(5)
        for numclusts in range(1, 6):
            T = KMeans(n_clusters=numclusts, n_init=5).fit(POINTS[::5])
            silh = self.inter_vs_intra_dist(POINTS[::5], T.labels_)
            sil[numclusts - 1] = np.mean(silh)
        numclusters = np.argmax(sil) + 1
        T = KMeans(n_clusters=numclusters, n_init=5).fit(POINTS)
        medianvalues = np.array([np.median(POINTS[T.labels_ == i], axis=0) for i in range(numclusters)])
        fixationcluster = np.argmin(np.sum(medianvalues[:, 1:3], axis=1))
        T.labels_[T.labels_ == fixationcluster] = 100
        fixationcluster2 = np.where((medianvalues[:, 1] < medianvalues[fixationcluster, 1] +
                                     3 * np.std(POINTS[T.labels_ == fixationcluster][:, 1])) &
                                    (medianvalues[:, 2] < medianvalues[fixationcluster, 2] +
                                     3 * np.std(POINTS[T.labels_ == fixationcluster][:, 2])))[0]
        fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
        for cluster in fixationcluster2:
            T.labels_[T.labels_ == cluster] = 100
        T.labels_[T.labels_ != 100] = 2
        T.labels_[T.labels_ == 100] = 1
        return altind[T.labels_ == 2]

    def remove_not_fixations(self, fixationindexes, notfixations):
        fixationindexes = np.setdiff1d(fixationindexes, notfixations)
        return fixationindexes

    def classify_saccades(self, fixationindexes, points):
        saccadeindexes = np.setdiff1d(np.arange(len(points)), fixationindexes)
        saccadetimes = self.find_behavioral_times(saccadeindexes)
        return saccadeindexes, saccadetimes

    def round_times(self, fixationtimes, saccadetimes):
        round5 = np.mod(fixationtimes, self.samprate * 1000)
        round5[0, round5[0] > 0] = self.samprate * 1000 - round5[0, round5[0] > 0]
        round5[1] = -round5[1]
        fixationtimes = np.round((fixationtimes + round5) / (self.samprate * 1000)).astype(int)
        fixationtimes[fixationtimes < 1] = 1

        round5 = np.mod(saccadetimes, self.samprate * 1000)
        round5[0] = -round5[0]
        round5[1, round5[1] > 0] = self.samprate * 1000 - round5[1, round5[1] > 0]
        saccadetimes = np.round((saccadetimes + round5) / (self.samprate * 1000)).astype(int)
        saccadetimes[saccadetimes < 1] = 1

        return fixationtimes, saccadetimes

    def calculate_cluster_values(self, fixationtimes, saccadetimes, eyedat):
        x = eyedat[0]
        y = eyedat[1]
        pointfix = [self.extract_variables(x[fix[0]:fix[1]], y[fix[0]:fix[1]]) for fix in fixationtimes.T]
        pointsac = [self.extract_variables(x[sac[0]:sac[1]], y[sac[0]:sac[1]]) for sac in saccadetimes.T]
        recalc_meanvalues = [np.nanmean(pointfix, axis=0), np.nanmean(pointsac, axis=0)]
        recalc_stdvalues = [np.nanstd(pointfix, axis=0), np.nanstd(pointsac, axis=0)]
        return pointfix, pointsac, recalc_meanvalues, recalc_stdvalues

    def extract_fixations(self, fixationtimes, eyedat):
        x = eyedat[0]
        y = eyedat[1]
        fixations = [np.mean([x[fix[0]:fix[1]], y[fix[0]:fix[1]]], axis=1) for fix in fixationtimes.T]
        return np.array(fixations)

    def extract_variables(self, xss, yss):
        if len(xss) < 3:
            return np.full(6, np.nan)
        vel = np.sqrt(np.diff(xss) ** 2 + np.diff(yss) ** 2) / self.samprate
        angle = np.degrees(np.arctan2(np.diff(yss), np.diff(xss)))
        accel = np.abs(np.diff(vel)) / self.samprate
        dist = [np.sqrt((xss[a] - xss[a + 2]) ** 2 + (yss[a] - yss[a + 2]) ** 2) for a in range(len(xss) - 2)]
        rot = [np.abs(angle[a] - angle[a + 1]) for a in range(len(xss) - 2)]
        rot = [r if r <= 180 else 360 - r for r in rot]
        return [np.max(vel), np.max(accel), np.mean(dist), np.mean(vel), np.abs(np.mean(angle)), np.mean(rot)]

    def inter_vs_intra_dist(self, X, labels):
        n = len(labels)
        k = len(np.unique(labels))
        count = np.bincount(labels)
        mbrs = (np.arange(k) == labels[:, None])
        avgDWithin = np.full(n, np.inf)
        avgDBetween = np.full((n, k), np.inf)
        for j in range(n):
            distj = np.sum((X - X[j]) ** 2, axis=1)
            for i in range(k):
                if i == labels[j]:
                    avgDWithin[j] = np.sum(distj[mbrs[:, i]]) / max(count[i] - 1, 1)
                else:
                    avgDBetween[j, i] = np.sum(distj[mbrs[:, i]]) / count[i]
        minavgDBetween = np.min(avgDBetween, axis=1)
        silh = (minavgDBetween - avgDWithin) / np.maximum(avgDWithin, minavgDBetween)
        return silh