import numpy as np
import scipy.signal as signal
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def detect_fixations_saccades(positions):
    """
    Detect fixations and saccades using k-means clustering.
    :param positions: 2D numpy array with shape (N, 2) containing [x, y] eye position data.
    :return: fixation_start_stop and saccade_start_stop as numpy arrays of shape (M,2) or empty if no events detected.
    """
    if positions.shape[0] < 500:
        logging.info("Insufficient data points (< 500), returning empty arrays.")
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)
    
    # Apply low-pass filtering to smooth position signals
    logger.info("Preprocessing positions data for fixation detection")
    x, y = _apply_lowpass_filter(positions)
    pdb.set_trace()
    # Compute velocity, acceleration, angular velocity, and displacement
    logger.info("Extracting motion parameters for k-means clustering")
    feature_matrix = _compute_motion_features(x, y)
    
    # Normalize features for clustering
    logger.info("Normalizing parameters for k-means clustering")
    feature_matrix = _normalize_features(feature_matrix)
    
    # Get initial estimate of fixation indices through global k means clustering of the data
    logger.info("Performing global clustering of points for 2 to 5 cluster sizes")
    fixation_indices = _extract_fixation_indices_through_global_k_means(feature_matrix)
    
    # Extract start-stop intervals for fixations
    fixation_start_stop = _extract_behavior_intervals(fixation_indices)
    
    # Perform local re-clustering for refinement
    logger.info("Refining fixation start-stop indices using local reclustering")
    not_fixations = _refine_fixation_classification_to_get_notfix_inds(fixation_start_stop, feature_matrix)
    fixation_indices = np.setdiff1d(fixation_indices, not_fixations)
    saccade_indices = np.setdiff1d(np.arange(len(feature_matrix)), fixation_indices)
    
    # Further filtering to remove short fixations and saccades
    fixation_start_stop = _extract_behavior_intervals(fixation_indices + 1) # The +1 is to account for the smaller vel/accel vectors compared to eyedat
    fixation_start_stop = _filter_behavior_intervals(fixation_start_stop, min_duration=25)
    saccade_start_stop = _extract_behavior_intervals(saccade_indices + 1)
    saccade_start_stop = _filter_behavior_intervals(saccade_start_stop, min_duration=10)
    
    # Ensure output arrays have the correct shape
    fixation_start_stop = fixation_start_stop if fixation_start_stop.size else np.empty((0, 2), dtype=int)
    saccade_start_stop = saccade_start_stop if saccade_start_stop.size else np.empty((0, 2), dtype=int)
    
    return fixation_start_stop, saccade_start_stop


def _apply_lowpass_filter(positions):
    """Applies a low-pass FIR filter to smooth eye movement data."""
    fltord = 60  # Filter order
    lowpass_freq = 30  # Cutoff frequency in Hz
    nyquist_freq = 500  # Nyquist frequency for 1000Hz sampling rate
    flt = signal.firwin2(fltord, 
                         [0, lowpasfrq / nyqfrq, lowpasfrq / nyqfrq, 1], 
                         [1, 1, 0, 0])
    buffer = int(100 / (samprate * 1000))
    x = np.pad(positions[:, 0], (buffer, buffer), 'reflect')
    y = np.pad(positions[:, 1], (buffer, buffer), 'reflect')
    flt = signal.firwin(fltord, cutoff=lowpass_freq/nyquist_freq, pass_zero=True)
    x = signal.filtfilt(flt, 1, x)
    y = signal.filtfilt(flt, 1, y)
    x = x[buffer:-buffer]
    y = y[buffer:-buffer]
    return x, y


def _compute_motion_features(xss, yss):
    """Computes velocity, acceleration, angular velocity, and displacement from smoothed position data."""
    velx, vely = np.diff(xss), np.diff(yss)
    vel = np.sqrt(velx**2 + vely**2)
    accel = np.abs(np.diff(vel))
    angle = np.degrees(np.arctan2(vely, velx))[:-1]
    vel = vel[:-1]
    rot, displacement = np.zeros(len(xss)-2), np.zeros(len(xss)-2)
    for i in range(len(xss)-2):
        rot[i] = np.abs(angle[i] - angle[i+1])
        displacement[i] = np.sqrt((xss[i] - xss[i+2])**2 + (yss[i] - yss[i+2])**2)
    rot[rot > 180] -= 180  # Normalize angular velocity to be within 180 degrees
    rot = 360 - rot
    return np.column_stack((displacement, vel, accel, rot))


def _normalize_features(feature_matrix):
    """
    Normalize feature values to [0,1] range to ensure even scaling across features.
    """
    for i in range(feature_matrix.shape[1]):
        threshold = np.mean(feature_matrix[:, i]) + 3 * np.std(feature_matrix[:, i])
        feature_matrix[feature_matrix[:, i] > threshold, i] = threshold
        feature_matrix[:, i] -= np.min(feature_matrix[:, i])
        feature_matrix[:, i] /= np.max(feature_matrix[:, i])
    return feature_matrix


def _extract_fixation_indices_through_global_k_means(feature_matrix):
    """
    Perform global clustering to classify fixations and saccades using K-Means.
    Parameters:
    feature_matrix (np.ndarray): Feature matrix where each row represents a data point.
    Returns:
    tuple: (fixation_indices, saccade_indices) where each is an array of indices.
    """
    # Determine the optimal number of clusters
    num_clusters = __determine_optimal_clusters(feature_matrix[:, 1:4])
    labels = KMeans(n_clusters=num_clusters, n_init=5).fit_predict(feature_matrix)
    # Compute mean and standard deviation for each cluster
    unique_clusters = np.unique(labels)
    mean_values = np.array([np.mean(feature_matrix[labels == k], axis=0) for k in unique_clusters])
    std_values = np.array([np.std(feature_matrix[labels == k], axis=0) for k in unique_clusters])
    # Identify primary and secondary fixation clusters
    fixation_cluster = np.argmin(np.sum(mean_values[:, 1:3], axis=1))
    labels[labels == fixation_cluster] = 100
    secondary_fixation_clusters = np.where(
        mean_values[:, 1] < mean_values[fixation_cluster, 1] + 3 * std_values[fixation_cluster, 1]
    )[0]
    labels[np.isin(labels, secondary_fixation_clusters)] = 100
    labels[labels != 100] = 2
    labels[labels == 100] = 1
    fixation_indices = np.where(labels == 1)[0]
    return fixation_indices

def __determine_optimal_clusters(data):
    """
    Determine the optimal number of clusters using silhouette scoring.
    """
    best_sil, best_k = -1, 2
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, n_init=5).fit(data)
        sil_score = silhouette_score(data, kmeans.labels_)
        if sil_score > best_sil:
            best_sil, best_k = sil_score, k
    return best_k


def _refine_fixation_classification_to_get_notfix_inds(fixation_start_stop, feature_matrix):
    """
    Refines the classification of fixation and non-fixation points based on clustering analysis.
    Identifies non-fixation indices by re-evaluating local fixation clusters.
    
    :param fixation_start_stop: (N,2) numpy array where each row contains [start, stop] indices of fixations.
    :param feature_matrix: (M,4) numpy array containing computed features: [displacement, velocity, acceleration, angular velocity].
    :return: numpy array containing indices of non-fixation points.
    """
    non_fixation_indices = []
    # Iterate through each fixation period
    for i in range(fixation_start_stop.shape[0]):
        # Select an extended range around the fixation (50 samples before and after)
        surrounding_indices = np.arange(fixation_start_stop[i, 0] - 50, fixation_start_stop[i, 1] + 50)
        surrounding_indices = surrounding_indices[(surrounding_indices >= 0) & (surrounding_indices < len(feature_matrix))]
        local_features = feature_matrix[surrounding_indices, :]
        # Determine optimal number of clusters using silhouette score
        silhouette_scores = []
        for num_clusters in range(1, 6):  # Testing between 1 to 5 clusters
            kmeans = KMeans(n_clusters=num_clusters, n_init=5, random_state=42)
            labels = kmeans.fit_predict(local_features[::5, :])  # Use every 5th data point to reduce computation
            silhouette_scores.append(silhouette_score(local_features[::5, :], labels))
        # Select optimal number of clusters based on highest silhouette score
        silhouette_scores = np.array(silhouette_scores)
        silhouette_scores[silhouette_scores > 0.9 * np.max(silhouette_scores)] = 1
        optimal_clusters = np.argmax(silhouette_scores) + 1
        # Perform KMeans clustering with the chosen optimal cluster count
        kmeans = KMeans(n_clusters=optimal_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(local_features)
        # Compute median feature values and range for each cluster
        cluster_medians = np.median([local_features[cluster_labels == k] for k in range(optimal_clusters)], axis=1)
        cluster_ranges = np.array([
            [np.max(local_features[cluster_labels == k], axis=0), np.min(local_features[cluster_labels == k], axis=0)]
            if np.any(cluster_labels == k) else np.ones((2, local_features.shape[1]))
            for k in range(optimal_clusters)
        ])
        # Identify fixation cluster as the one with minimum velocity and acceleration
        primary_fixation_cluster = np.argmin(np.sum(cluster_medians[:, 1:3], axis=1))
        cluster_labels[cluster_labels == primary_fixation_cluster] = 100
        # Identify additional fixation clusters whose velocity and acceleration lie within the range of the primary fixation cluster
        secondary_fixation_clusters = np.where(
            (cluster_ranges[:, 0, 1] > cluster_medians[primary_fixation_cluster, 1]) &  # Velocity range upper bound
            (cluster_ranges[:, 1, 1] < cluster_medians[primary_fixation_cluster, 1]) &  # Velocity range lower bound
            (cluster_ranges[:, 0, 2] > cluster_medians[primary_fixation_cluster, 2]) &  # Acceleration range upper bound
            (cluster_ranges[:, 1, 2] < cluster_medians[primary_fixation_cluster, 2])    # Acceleration range lower bound
        )[0]
        # Exclude the primary fixation cluster from the secondary list
        secondary_fixation_clusters = secondary_fixation_clusters[secondary_fixation_clusters != primary_fixation_cluster]
        # Mark all identified fixation clusters
        for fixation_cluster in secondary_fixation_clusters:
            cluster_labels[cluster_labels == fixation_cluster] = 100
        # Assign non-fixation clusters as 2 and fixation clusters as 1
        cluster_labels[cluster_labels != 100] = 2
        cluster_labels[cluster_labels == 100] = 1
        # Collect indices of non-fixation points
        non_fixation_indices.extend(surrounding_indices[cluster_labels == 2])
    return np.array(non_fixation_indices)


def _extract_behavior_intervals(indices):
    """
    Convert classified index sequences into continuous behavioral periods.
    """
    if len(indices) == 0:
        return np.empty((0, 2), dtype=int)
    diffs = np.diff(indices)
    gaps = np.where(diffs > 1)[0]
    times = np.vstack((indices[np.insert(gaps + 1, 0, 0)], indices[np.append(gaps, len(indices) - 1)])).T
    pdb.set_trace()
    return times


def _filter_behavior_intervals(intervals, min_duration):
    """
    Remove behavioral intervals that are shorter than the minimum duration.
    """
    if intervals.size == 0:
        return intervals
    return intervals[np.where((intervals[:, 1] - intervals[:, 0]) >= min_duration)]
