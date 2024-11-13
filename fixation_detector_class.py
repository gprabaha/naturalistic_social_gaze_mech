import numpy as np
import logging
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import gc

import util

import pdb


class FixationDetector:
    def __init__(self, session_name = None, samprate=1/1000, params=None, num_cpus=1):
        self.setup_logger()
        self.session_name = session_name
        self.params = params
        self.samprate = samprate
        self.num_cpus = num_cpus
        self.use_parallel = params['use_parallel']
        self.variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
        self.fltord = 60
        self.lowpasfrq = 30
        self.nyqfrq = 1000 / 2
        self.flt = signal.firwin2(self.fltord, [0, self.lowpasfrq / self.nyqfrq, self.lowpasfrq / self.nyqfrq, 1], [1, 1, 0, 0])
        self.buffer = int(100 / (self.samprate * 1000))
        self.fixationstats = []


    def setup_logger(self):
        """Sets up the logger for the class."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)


    def detect_fixations(self, eyedat):
        """Detects fixations in the provided eye data.
        Args:
            eyedat (list): List containing x and y coordinates of eye data.
        Returns:
            dict: Fixation statistics including fixation indices and times.
        """
        if not eyedat:
            self.logger.error("No data file found")
            raise ValueError("No data file found")
        self.fixationstats = self.process_eyedat(eyedat)
        return self.fixationstats


    def process_eyedat(self, data):
        """Processes the eye data to detect fixations, removes outliers and adjusts
        the fixation indices by truncating non-fixation periods.
        Args:
            data (list): List containing x and y coordinates of eye data.
        Returns:
            dict: A dictionary containing detected fixation indices, times, and outlier info.
        """
        if len(data[0]) > int(30 / (self.samprate * 1000)):
            self.logger.debug("Preprocessing data")
            x, y = self.preprocess_data(data)
            self.logger.debug("Extracting parameters")
            vel, accel, angle, dist, rot = self.extract_parameters(x, y)
            points = self.normalize_parameters(dist, vel, accel, rot)  # points are now normalized
            self.logger.debug("Detecting fixations")
            T, meanvalues, stdvalues = self.global_clustering(points)
            fixationcluster, fixationcluster2 = self.find_fixation_clusters(meanvalues, stdvalues)
            T = self.classify_fixations(T, fixationcluster, fixationcluster2)
            # Find fixation start indices and ranges
            _, fixation_indices = self.behavioral_start_index_and_range(T, 1)
            # Perform local reclustering to remove non-fixations
            notfixations = self.local_reclustering((fixation_indices, points, x, y))
            # Truncate or split fixations based on the non-fixation points
            fixation_indices = self.remove_not_fixations(fixation_indices, notfixations)
            # Apply duration threshold after removing non-fixations
            _, fixation_indices = self.apply_duration_threshold(fixation_indices, int(0.025 / self.samprate))
            # Logging how many fixations and outliers were found
            num_fixations = fixation_indices.shape[1] if fixation_indices.size > 0 else 0
            self.logger.info(f"Final number of fixations identified: {num_fixations}")
            return {
                'session_name': self.session_name,
                'fixationindices': util.reshape_to_ensure_data_rows_represent_samples(fixation_indices),
                'fixationtimes': util.reshape_to_ensure_data_rows_represent_samples(fixation_indices) * self.samprate,
                'XY': np.array([x, y]),
                'variables': self.variables
            }
        else:
            self.logger.warning("Data too short for processing")
            return {
                'fixationindices': [],
                'XY': np.array([data[0], data[1]]),
                'variables': self.variables
            }


    def preprocess_data(self, eyedat):
        """Pads, resamples, and filters the eye data.
        Args:
            eyedat (list): List containing x and y coordinates of eye data.
        Returns:
            tuple: Preprocessed x and y coordinates.
        """
        self.logger.debug("Preprocessing x and y data")
        x = np.pad(eyedat[0], (self.buffer, self.buffer), 'reflect')
        y = np.pad(eyedat[1], (self.buffer, self.buffer), 'reflect')
        x = self.resample_data(x)
        y = self.resample_data(y)
        x = self.apply_filter(x)
        y = self.apply_filter(y)
        x = x[self.buffer:-self.buffer]
        y = y[self.buffer:-self.buffer]
        return x, y


    def resample_data(self, data):
        """Resamples the data based on the sampling rate.
        Args:
            data (np.ndarray): Array of eye data.
        Returns:
            np.ndarray: Resampled data.
        """
        self.logger.debug("Resampling data")
        t_old = np.linspace(0, len(data) - 1, len(data))
        resample_factor = self.samprate * 1000
        if resample_factor > 1:
            self.logger.error(f"Resample factor is too large: {resample_factor}")
            raise ValueError("Resample factor is too large, leading to excessive memory usage.")
        t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
        f = interp1d(t_old, data, kind='linear')
        return f(t_new)


    def apply_filter(self, data):
        """Applies a low-pass filter to the data.
        Args:
            data (np.ndarray): Array of data to be filtered.
        Returns:
            np.ndarray: Filtered data.
        """
        self.logger.debug("Applying filter to data")
        return signal.filtfilt(self.flt, 1, data)


    def extract_parameters(self, x, y):
        """Extracts velocity, acceleration, angle, distance, and rotation parameters from eye data.
        Args:
            x (np.ndarray): x-coordinates of eye data.
            y (np.ndarray): y-coordinates of eye data.
        Returns:
            tuple: Extracted parameters - velocity, acceleration, angle, distance, and rotation.
        """
        self.logger.debug("Extracting parameters from x and y data")
        velx = np.diff(x)
        vely = np.diff(y)
        vel = np.sqrt(velx ** 2 + vely ** 2)
        accel = np.abs(np.diff(vel))
        angle = np.degrees(np.arctan2(vely, velx))
        vel = vel[:-1]  # Synchronize length
        rot = np.zeros(len(x) - 2)
        dist = np.zeros(len(x) - 2)
        for a in range(len(x) - 2):
            rot[a] = np.abs(angle[a] - angle[a + 1])
            dist[a] = np.sqrt((x[a] - x[a + 2]) ** 2 + (y[a] - y[a + 2]) ** 2)
        rot[rot > 180] = 360 - rot[rot > 180]  # Ensure rotation is within [0, 180]
        return vel, accel, angle, dist, rot


    def normalize_parameters(self, dist, vel, accel, rot):
        """Normalizes the extracted parameters to the range [0, 1].
        Args:
            dist (np.ndarray): Distance array.
            vel (np.ndarray): Velocity array.
            accel (np.ndarray): Acceleration array.
            rot (np.ndarray): Rotation array.
        Returns:
            np.ndarray: Normalized parameters.
        """
        self.logger.debug("Normalizing parameters")
        points = np.stack([dist, vel, accel, rot], axis=1)
        for ii in range(points.shape[1]):
            points[:, ii] = np.clip(points[:, ii], None, np.mean(points[:, ii]) + 3 * np.std(points[:, ii]))
            points[:, ii] = (points[:, ii] - np.min(points[:, ii])) / (np.max(points[:, ii]) - np.min(points[:, ii]))
        return points


    def global_clustering(self, points):
        """Performs global clustering on the normalized parameters.
        Args:
            points (np.ndarray): Normalized parameters.
        Returns:
            tuple: Cluster labels, mean values, and standard deviations.
        """
        self.logger.info("Starting global_clustering...")
        numclusts_range = list(range(2, 6))
        sil = np.zeros(5)
        results = []
        if self.use_parallel:
            self.logger.info("Using parallel processing for global clustering")
            results = self.parallel_global_clustering(points, numclusts_range, min(len(numclusts_range), self.num_cpus))
        else:
            self.logger.info("Using serial processing for global clustering")
            for numclusts in tqdm(numclusts_range, desc="Serial Global Clustering Progress"):
                try:
                    result = self.cluster_and_silhouette(points, numclusts)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing numclusts {numclusts}: {e}")
        for numclusts, score in results:
            sil[numclusts - 2] = score
        numclusters = np.argmax(sil) + 2
        self.logger.info(f"Optimal number of clusters: {numclusters}")
        T = KMeans(n_clusters=numclusters, n_init=5).fit(points)
        labels = T.labels_
        meanvalues = np.array([np.mean(points[labels == i], axis=0) for i in range(numclusters)])
        stdvalues = np.array([np.std(points[labels == i], axis=0) for i in range(numclusters)])
        return labels, meanvalues, stdvalues


    def parallel_global_clustering(self, points, numclusts_range, max_workers):
        """Performs global clustering using parallel processing.
        Args:
            points (np.ndarray): Normalized parameters.
            numclusts_range (list): List of cluster numbers to try.
            max_workers (int): Maximum number of parallel workers.
        Returns:
            list: Clustering results for each number of clusters.
        """
        self.logger.info("Using parallel processing with multiprocessing.Pool")
        with Pool(processes=max_workers) as pool:
            args = [(points, numclusts) for numclusts in numclusts_range]
            results = list(
                tqdm(
                    pool.imap_unordered(self.cluster_and_silhouette_wrapper, args),
                    total=len(numclusts_range),
                    desc="Parallel Global Clustering Progress"))
            pool.close()
            pool.join()
        return results


    def cluster_and_silhouette_wrapper(self, args):
        """Wrapper function for clustering and silhouette calculation.
        Args:
            args (tuple): Tuple containing points and number of clusters.
        Returns:
            tuple: Number of clusters and mean silhouette score.
        """
        points, numclusts = args
        return self.cluster_and_silhouette(points, numclusts)


    def cluster_and_silhouette(self, points, numclusts):
        """Performs clustering and calculates silhouette scores.
        Args:
            points (np.ndarray): Normalized parameters.
            numclusts (int): Number of clusters.
        Returns:
            tuple: Number of clusters and mean silhouette score.
        """
        self.logger.debug(f"Clustering with {numclusts} clusters")
        T = KMeans(n_clusters=numclusts, n_init=5).fit(points[::10, 1:4])
        silh = self.inter_vs_intra_dist(points[::10, 1:4], T.labels_)
        return numclusts, np.mean(silh)


    def find_fixation_clusters(self, meanvalues, stdvalues):
        """Identifies clusters that correspond to fixations.
        Args:
            meanvalues (np.ndarray): Mean values of clusters.
            stdvalues (np.ndarray): Standard deviations of clusters.
        Returns:
            tuple: Primary fixation cluster and additional fixation clusters.
        """
        self.logger.debug("Finding fixation clusters")
        fixationcluster = np.argmin(np.sum(meanvalues[:, 1:3], axis=1))
        fixationcluster2 = np.where(meanvalues[:, 1] < meanvalues[fixationcluster, 1] + 3 * stdvalues[fixationcluster, 1])[0]
        fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
        return fixationcluster, fixationcluster2


    def classify_fixations(self, T, fixationcluster, fixationcluster2):
        """Classifies clusters into fixation and non-fixation categories.
        Args:
            T (np.ndarray): Cluster labels.
            fixationcluster (int): Primary fixation cluster index.
            fixationcluster2 (np.ndarray): Additional fixation cluster indices.
        Returns:
            np.ndarray: Updated cluster labels.
        """
        self.logger.debug("Classifying fixations")
        T[T == fixationcluster] = 100
        for cluster in fixationcluster2:
            T[T == cluster] = 100
        T[T != 100] = 2
        T[T == 100] = 1
        return T


    def behavioral_start_index_and_range(self, T, label):
        """Finds start indices and range of behavioral events (fixations).
        Args:
            T (np.ndarray): Cluster labels.
            label (int): Label indicating fixation clusters.
        Returns:
            tuple: Start indices and event indices.
        """
        self.logger.debug("Finding behavioral start indices and ranges")
        start_inds = np.where(T == label)[0]
        return start_inds, self.find_event_indices(start_inds)


    def find_event_indices(self, start_inds):
        """Finds indices of events (fixations).
        Args:
            start_inds (np.ndarray): Start indices of events.
        Returns:
            np.ndarray: Event indices as start and end pairs.
        """
        self.logger.debug("Finding event indices")
        if start_inds.size == 0:
            return np.array([], dtype=int).reshape(2, 0)
        dind = np.diff(start_inds)
        gaps = np.where(dind > 1)[0]
        if gaps.size > 0:
            behaveind = np.split(start_inds, gaps + 1)
        else:
            behaveind = [start_inds]
        behaviortime = np.zeros((2, len(behaveind)), dtype=int)
        for i, ind in enumerate(behaveind):
            behaviortime[:, i] = [ind[0], ind[-1]]
        return behaviortime


    def apply_duration_threshold(self, indices, threshold):
        """Applies a duration threshold to filter short events.
        Args:
            times (np.ndarray): Array of event start and end indices.
            threshold (int): Duration threshold.
        Returns:
            np.ndarray: Filtered event indices.
        """
        self.logger.debug(f"Applying duration threshold: {threshold}")
        fixation_indices = indices[:, np.diff(indices, axis=0)[0] >= threshold]
        return fixation_indices[0, :], fixation_indices


    def local_reclustering(self, data):
        """Performs local reclustering to refine fixation detection.
        Args:
            data (tuple): Tuple containing fixation times and normalized parameters.
        Returns:
            np.ndarray: Indices of non-fixation events detected in reclustering.
        """
        self.logger.info("Starting local_reclustering...")
        fixation_indices, points, x, y = data
        num_fixes = fixation_indices.shape[1]
        notfixations = []
        do_parallel = self.params.get('do_local_reclustering_in_parallel', False)
        max_processes = min(16, self.num_cpus, num_fixes)
        if do_parallel:
            self.logger.info("Parallelizing local_reclustering over detected fix time points")
            with Pool(processes=max_processes) as pool:
                # Prepare the arguments as a tuple of (fix, points, x, y)
                args = [(fixation_indices[:, i], points, x, y) for i in range(num_fixes)]
                # Use imap_unordered for parallel processing
                results = list(
                    tqdm(
                        pool.imap_unordered(self.process_fixation_local_reclustering_wrapper, args),
                        total=num_fixes,
                        desc="Parallel Reclustering Progress"))
                pool.close()
                pool.join()
            if results:
                notfixations = np.concatenate(results)
            gc.collect()  # Ensure garbage collection after pool shutdown
        else:
            self.logger.info("Serial local_reclustering over detected fix time points")
            for i in tqdm(range(num_fixes), desc="Serial Reclustering Progress"):
                try:
                    result = self.process_fixation_local_reclustering(fixation_indices[:, i], points, x, y)
                    notfixations.append(result)
                except Exception as e:
                    self.logger.exception("Exception occurred during serial local reclustering")
            if notfixations:
                notfixations = np.concatenate(notfixations)
        self.logger.info("Finished local_reclustering...")
        return notfixations


    def process_fixation_local_reclustering_wrapper(self, args):
        """Wrapper function for processing fixation reclustering.
        Args:
            args (tuple): Tuple containing fixation indices, normalized parameters (velocity, acceleration, etc.), and x, y coordinates.
        Returns:
            np.ndarray: Indices of non-fixation events detected in reclustering.
        """
        fix, points, x, y = args
        return self.process_fixation_local_reclustering(fix, points, x, y)


    def process_fixation_local_reclustering(self, fix, points, x, y):
        """Processes local reclustering for a single fixation event with original reclustering followed by centroid-based boundary exclusion.
        Args:
            fix (np.ndarray): Fixation indices.
            points (np.ndarray): Normalized parameters (velocity, acceleration, etc.).
            x (np.ndarray): x-coordinates corresponding to the original eye data.
            y (np.ndarray): y-coordinates corresponding to the original eye data.
        Returns:
            np.ndarray: Indices of non-fixation events detected in reclustering.
        """
        altind = np.arange(fix[0] - 50, fix[1] + 50)
        altind = altind[(altind >= 0) & (altind < len(points))]
        POINTS = points[altind]
        x_relevant = x[altind]
        y_relevant = y[altind]
        # Original reclustering logic
        numclusts_range = range(1, 6)
        sil_results = [self.compute_sil((numclusts, POINTS)) for numclusts in numclusts_range]
        sil = np.zeros(5)
        for mean_sil, numclusts in sil_results:
            sil[numclusts - 1] = mean_sil
        numclusters = np.argmax(sil) + 1
        # KMeans clustering on normalized parameters (velocity, acceleration, etc.)
        T = KMeans(n_clusters=numclusters, n_init=5).fit(POINTS)
        # Median values of the clusters
        medianvalues = np.array([np.median(POINTS[T.labels_ == i], axis=0) for i in range(numclusters)])
        # Find the fixation cluster (typically the one with the smallest velocity and acceleration)
        fixationcluster = np.argmin(np.sum(medianvalues[:, 1:3], axis=1))  # Velocity and acceleration
        fixation_indices = np.where(T.labels_ == fixationcluster)[0]
        if fixation_indices.size == 0:
            self.logger.warning("No points assigned to the fixation cluster")
            return np.array([])  # Return an empty array if no valid fixations
        # Mark the fixation cluster
        T.labels_[fixation_indices] = 100
        # Find additional fixation clusters based on velocity and acceleration thresholds
        fixationcluster2 = np.where(
            (medianvalues[:, 1] < medianvalues[fixationcluster, 1] + 3 * np.std(POINTS[fixation_indices][:, 1])) &
            (medianvalues[:, 2] < medianvalues[fixationcluster, 2] + 3 * np.std(POINTS[fixation_indices][:, 2]))
        )[0]
        fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
        # Mark all fixation clusters
        for cluster in fixationcluster2:
            cluster_indices = np.where(T.labels_ == cluster)[0]
            T.labels_[cluster_indices] = 100
        # Classify all non-fixations as label 2, and fixations as label 1
        T.labels_[T.labels_ != 100] = 2
        T.labels_[T.labels_ == 100] = 1
        # Now, apply centroid-based boundary filtering after reclustering
        # Filter only on clusters marked as fixations (T.labels_ == 1)
        fixation_mask = (T.labels_ == 1)
        # Find islands of 1s (contiguous fixation clusters)
        fixation_islands = self.find_fixation_islands(fixation_mask)
        # Fetch pixel threshold for boundary outlier removal from params
        boundary_threshold = self.params.get('pixel_threshold_for_boundary_outlier_removal', 50)
        # Calculate boundary limits based on the available x and y data
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        # Track the non-fixation points
        non_fixation_points = []
        # Check each fixation island separately
        for island in fixation_islands:
            # Calculate the centroid of the current fixation island
            centroid_x = np.mean(x_relevant[island])
            centroid_y = np.mean(y_relevant[island])
            # Check if the centroid is too close to the boundaries
            if (centroid_x < (x_min + boundary_threshold)) or (centroid_x > (x_max - boundary_threshold)) or \
            (centroid_y < (y_min + boundary_threshold)) or (centroid_y > (y_max - boundary_threshold)):
                # Mark this fixation island as non-fixation
                non_fixation_points.extend(island)
        # Return the indices of non-fixation points
        return altind[non_fixation_points]


    def find_fixation_islands(self, fixation_mask):
        """Finds contiguous islands of fixation points (1s) in the fixation mask.
        Args:
            fixation_mask (np.ndarray): Boolean array indicating fixation points (1s).
        Returns:
            list: List of arrays, where each array is a contiguous set of fixation points.
        """
        islands = []
        indices = np.where(fixation_mask)[0]
        if indices.size == 0:
            return islands
        # Group contiguous indices into islands
        dind = np.diff(indices)
        gaps = np.where(dind > 1)[0]
        if gaps.size > 0:
            islands = np.split(indices, gaps + 1)
        else:
            islands = [indices]
        return islands


    def compute_sil(self, data):
        """Computes silhouette scores for a given number of clusters.
        Args:
            data (tuple): Tuple containing number of clusters and data points.
        Returns:
            tuple: Mean silhouette score and number of clusters.
        """
        try:
            numclusts, POINTS = data
            T = KMeans(n_clusters=numclusts, n_init=5).fit(POINTS[::5])
            silh = self.inter_vs_intra_dist(POINTS[::5], T.labels_)
            return np.nanmean(silh), numclusts
        except Exception as e:
            self.logger.exception("Exception occurred during silhouette computation")
            return 0.0, numclusts


    def remove_not_fixations(self, fixationindices, notfixations):
        """Truncates or splits fixation indices by removing periods marked as non-fixations.
        Args:
            fixationindices (np.ndarray): Array of start and end fixation indices.
            notfixations (np.ndarray): Array of non-fixation indices.
        Returns:
            np.ndarray: Updated fixation indices with non-fixation periods removed.
        """
        self.logger.debug("Truncating or splitting fixations based on non-fixation points")
        updated_fixationindices = []
        for start, end in fixationindices.T:
            # Find overlap between the fixation and the non-fixations
            non_fix_overlap = notfixations[(notfixations >= start) & (notfixations <= end)]
            if non_fix_overlap.size == 0:
                # No overlap, keep the fixation as is
                updated_fixationindices.append([start, end])
            else:
                # Split the fixation around the non-fixation periods
                valid_periods = self.split_fixation_on_non_fixation(start, end, non_fix_overlap)
                updated_fixationindices.extend(valid_periods)
        if updated_fixationindices:
            updated_fixationindices = np.array(updated_fixationindices).T
        else:
            updated_fixationindices = np.array([], dtype=int).reshape(2, 0)  # Return empty if no valid fixations
        return updated_fixationindices


    def split_fixation_on_non_fixation(self, start, end, non_fix_overlap):
        """Splits a fixation into valid periods based on non-fixation overlap.
        Args:
            start (int): Start index of the fixation.
            end (int): End index of the fixation.
            non_fix_overlap (np.ndarray): Indices of non-fixation points within the fixation.
        Returns:
            list: List of valid fixation start and end indices after splitting.
        """
        valid_periods = []
        last_start = start
        # Go through the non-fixation points and split the fixation
        for non_fix in non_fix_overlap:
            if last_start < non_fix - 1:
                # Add the valid period before the non-fixation point
                valid_periods.append([last_start, non_fix - 1])
            last_start = non_fix + 1  # Move start to after the non-fixation point
        # Add the last valid period after the final non-fixation point
        if last_start <= end:
            valid_periods.append([last_start, end])
        return valid_periods


    def inter_vs_intra_dist(self, X, labels):
        """Calculates silhouette scores for clustering.
        Args:
            X (np.ndarray): Data points.
            labels (np.ndarray): Cluster labels.
        Returns:
            np.ndarray: Silhouette scores for each point.
        """
        n = len(labels)
        k = len(np.unique(labels))
        count = np.bincount(labels)
        if k == 0 or n == 0:
            self.logger.error("Error: Labels or data points are empty.")
            return np.full(n, 0.0)  # Return an array of zeros
        if np.any(count == 0):
            self.logger.error("Error: One or more clusters have no members.")
            return np.full(n, 0.0)  # Return an array of zeros
        try:
            mbrs = (np.arange(k) == labels[:, None])
            avgDWithin = np.full(n, np.inf)
            avgDBetween = np.full((n, k), np.inf)
            for j in range(n):
                distj = np.sum((X - X[j]) ** 2, axis=1)
                for i in range(k):
                    if i == labels[j]:
                        if count[i] > 1:
                            avgDWithin[j] = np.sum(distj[mbrs[:, i]]) / (count[i] - 1)
                        else:
                            avgDWithin[j] = np.nan  # Avoid division by zero, set to NaN
                    else:
                        if count[i] > 0:
                            avgDBetween[j, i] = np.sum(distj[mbrs[:, i]]) / count[i]
                        else:
                            avgDBetween[j, i] = np.nan  # Avoid division by zero, set to NaN
            minavgDBetween = np.nanmin(avgDBetween, axis=1)  # Use nanmin to ignore NaNs
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for invalid operations
                silh = (minavgDBetween - avgDWithin) / np.maximum(avgDWithin, minavgDBetween)
                silh = np.nan_to_num(silh, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs and infs with 0
        except Exception as e:
            self.logger.exception("Exception occurred during silhouette calculation")
            return np.full(n, 0.0)  # Return an array of zeros in case of an error
        return silh
    
