
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans


def detect_fixation_in_position_array(positions, session_name, samprate=1/1000):
    fix_params = _get_fixation_parameters(session_name, samprate)
    if len(positions[0]) > int(30 / (fix_params['samprate'] * 1000)):
        print("Preprocessing positions data for fixation detection")
        x, y = _preprocess_data(positions, fix_params)
        print("Extracting vel, accel, etc. parameters for k-means clustering")
        dist, vel, accel, rot = _extract_motion_parameters(x, y)
        print("Normalizing parameters for k-means clustering")
        normalized_data_params = _normalize_motion_parameters(dist, vel, accel, rot)
        clustering_labels, cluster_means, cluster_stds = _kmeans_cluster_all_points_globally(normalized_data_params)
        '''
        now that global clustering is done, we have to identify which clusters are fixations based on velocity
        and then we have to find other clusters within 3 sd of the fixation cluster. after that, the fixations
        will need to be labelled, and then for each consecutive points, we have to do a local reclustering like
        we have done before to eliminate points which are not fixations
        '''
    else:
        print("!! Data too short for fixation detectionprocessing !!")
        return {
            'fixationindices': [],
            'XY': np.array([positions[0], positions[1]])
        }



def _get_fixation_parameters(session_name=None, samprate=1/1000, num_cpus=1):
    # Initialize parameters
    variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
    fltord = 60
    lowpasfrq = 30
    nyqfrq = 1000 / 2  # Nyquist frequency
    flt = signal.firwin2(fltord, 
                         [0, lowpasfrq / nyqfrq, lowpasfrq / nyqfrq, 1], 
                         [1, 1, 0, 0])
    buffer = int(100 / (samprate * 1000))
    # Construct the parameters dictionary
    fixation_params = {
        'session_name': session_name,
        'samprate': samprate,
        'num_cpus': num_cpus,
        'variables': variables,
        'fltord': fltord,
        'lowpasfrq': lowpasfrq,
        'nyqfrq': nyqfrq,
        'flt': flt,
        'buffer': buffer
    }
    return fixation_params



def _preprocess_data(positions, fix_params):
    """Pads, resamples, and filters the eye data.
    Args:
        eyedat (list): List containing x and y coordinates of eye data.
    Returns:
        tuple: Preprocessed x and y coordinates.
    """
    print("Preprocessing x and y data")
    x = np.pad(positions[0], (fix_params['buffer'], fix_params['buffer']), 'reflect')
    y = np.pad(positions[1], (fix_params['buffer'], fix_params['buffer']), 'reflect')
    x = __resample_data(x, fix_params)
    y = __resample_data(y, fix_params)
    x = __apply_filter(x, fix_params)
    y = __apply_filter(y, fix_params)
    x = x[fix_params['buffer']:-fix_params['buffer']]
    y = y[fix_params['buffer']:-fix_params['buffer']]
    return x, y

def __resample_data(data, fix_params):
    """Resamples the data based on the sampling rate.
    Args:
        data (np.ndarray): Array of eye data.
    Returns:
        np.ndarray: Resampled data.
    """
    t_old = np.linspace(0, len(data) - 1, len(data))
    resample_factor = fix_params['samprate'] * 1000
    if resample_factor > 1:
        print(f"Resample factor is too large: {resample_factor}")
        raise ValueError("Resample factor is too large, leading to excessive memory usage.")
    t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
    f = interp1d(t_old, data, kind='linear')
    return f(t_new)

def __apply_filter(data, fix_params):
    """Applies a low-pass filter to the data.
    Args:
        data (np.ndarray): Array of data to be filtered.
    Returns:
        np.ndarray: Filtered data.
    """
    return signal.filtfilt(fix_params['flt'], 1, data)



def _extract_motion_parameters(x, y):
        """Extracts velocity, acceleration, angle, distance, and rotation parameters from eye data.
        Args:
            x (np.ndarray): x-coordinates of eye data.
            y (np.ndarray): y-coordinates of eye data.
        Returns:
            tuple: Extracted parameters - velocity, acceleration, angle, distance, and rotation.
        """
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
        return dist, vel, accel, rot



def _normalize_motion_parameters(dist, vel, accel, rot):
    """
    Normalizes the extracted motion parameters to a range of [0, 1] for clustering.
    Args:
        dist (np.ndarray): Distance parameter.
        vel (np.ndarray): Velocity parameter.
        accel (np.ndarray): Acceleration parameter.
        rot (np.ndarray): Rotation parameter.
    Returns:
        np.ndarray: A 2D array with normalized parameters.
    """
    # Stack the parameters into a single array
    parameters = np.stack([dist, vel, accel, rot], axis=1)
    # Normalize each parameter independently
    for i in range(parameters.shape[1]):
        # Clip values to mean + 3*std to reduce outlier influence
        max_val = np.mean(parameters[:, i]) + 3 * np.std(parameters[:, i])
        parameters[:, i] = np.clip(parameters[:, i], None, max_val)
        # Normalize to [0, 1]
        min_val = np.min(parameters[:, i])
        max_val = np.max(parameters[:, i])
        parameters[:, i] = (parameters[:, i] - min_val) / (max_val - min_val) if max_val > min_val else 0
    return parameters



def _kmeans_cluster_all_points_globally(normalized_data):
    """
    Performs global KMeans clustering with 2 to 5 clusters and selects the optimal configuration.
    Args:
        normalized_data (np.ndarray): Normalized motion parameters.
    Returns:
        tuple: 
            - clustering_labels (np.ndarray): Cluster labels for each data point.
            - cluster_means (np.ndarray): Mean values for each cluster.
            - cluster_stds (np.ndarray): Standard deviations for each cluster.
    """
    best_score = -1
    best_labels = None
    best_means = None
    best_stds = None
    for n_clusters in range(2, 6):  # Try 2 to 5 clusters
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=5).fit(normalized_data)
        # Evaluate clustering quality
        score = __evaluate_clustering_quality(normalized_data, kmeans.labels_)
        print(f"Number of clusters: {n_clusters}, Silhouette score: {score:.3f}")
        # Update best clustering based on silhouette score
        if score > best_score:
            best_score = score
            best_labels = kmeans.labels_
            best_means = np.array([np.mean(normalized_data[best_labels == i], axis=0) for i in range(n_clusters)])
            best_stds = np.array([np.std(normalized_data[best_labels == i], axis=0) for i in range(n_clusters)])
    print(f"Optimal number of clusters determined: {len(best_means)}")
    return best_labels, best_means, best_stds

def __evaluate_clustering_quality(data, labels):
    """
    Evaluates the quality of clustering using silhouette scores.
    Args:
        data (np.ndarray): The data points used for clustering.
        labels (np.ndarray): Cluster labels assigned to the data points.
    Returns:
        float: The average silhouette score across all data points.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1  # Invalid silhouette score for fewer than 2 clusters
    n_points = len(data)
    a_values = np.zeros(n_points)  # Intra-cluster distances
    b_values = np.full(n_points, np.inf)  # Inter-cluster distances
    # Compute distances between all points
    distances = np.linalg.norm(data[:, None] - data[None, :], axis=2)
    for label in unique_labels:
        # Points in the current cluster
        cluster_points = (labels == label)
        other_points = ~cluster_points
        # Intra-cluster distance (a): Mean distance to points in the same cluster
        cluster_distances = distances[cluster_points][:, cluster_points]
        np.fill_diagonal(cluster_distances, np.nan)  # Exclude self-distances
        a_values[cluster_points] = np.nanmean(cluster_distances, axis=1)
        # Inter-cluster distance (b): Mean distance to points in other clusters
        for other_label in unique_labels:
            if other_label != label:
                inter_distances = distances[cluster_points][:, labels == other_label]
                b_values[cluster_points] = np.minimum(b_values[cluster_points], np.mean(inter_distances, axis=1))
    # Silhouette score
    silhouette_scores = (b_values - a_values) / np.maximum(a_values, b_values)
    return np.nanmean(silhouette_scores)  # Average silhouette score
