
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pdb

def detect_fixation_in_position_array(positions, session_name, samprate=1/1000):
    fix_params = _get_fixation_parameters(session_name, samprate)
    if positions.shape[0] > int(30 / (fix_params['samprate'] * 1000)):
        print("Preprocessing positions data for fixation detection")
        x, y = _preprocess_data(positions, fix_params)
        print("Extracting vel, accel, etc. parameters for k-means clustering")
        dist, vel, accel, rot = _extract_motion_parameters(x, y)
        print("Normalizing parameters for k-means clustering")
        normalized_data_params = _normalize_motion_parameters(dist, vel, accel, rot)
        print("Performing global clustering of points for 2 to 5 cluster size")
        clustering_labels, cluster_means, cluster_stds = _kmeans_cluster_all_points_globally(normalized_data_params)
        print("Determining fixation cluster based on smallest mean velocity and additional clusters with velocity within 3sd velocity of fixation cluster")
        fixation_cluster, additional_fixation_clusters = _determine_fixation_clusters(cluster_means, cluster_stds)
        print("Updating point labels to fixation and not-fixation clusters based on previous analysis")
        fixation_labels = _classify_clusters_as_fixations_or_non_fixations(
            clustering_labels, fixation_cluster, additional_fixation_clusters)
        print("Calculating the start and stop indices of each fixation")
        fixation_start_stop_indices = _find_fixation_start_stop_indices(fixation_labels)
        return fixation_start_stop_indices
        '''
        now that global clustering is done, we have to identify which clusters are fixations based on velocity
        and then we have to find other clusters within 3 sd of the fixation cluster. after that, the fixations
        will need to be labelled, and then for each consecutive points, we have to do a local reclustering like
        we have done before to eliminate points which are not fixations. ensure that all stop indices are
        larger than the start indices but smaller than the position array size. throw error otherwise
        '''
    else:
        print("!! Data too short for fixation detection processing !!")
        return []



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
    x = np.pad(positions[:, 0], (fix_params['buffer'], fix_params['buffer']), 'reflect')
    y = np.pad(positions[:, 1], (fix_params['buffer'], fix_params['buffer']), 'reflect')
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
    Performs global KMeans clustering with 2 to 5 clusters and selects the optimal configuration 
    based on the silhouette score.
    The silhouette score measures the quality of clustering by comparing the distance between 
    each data point and points within the same cluster (intra-cluster distance) with the distance 
    between that data point and points in the nearest cluster (inter-cluster distance). 
    The silhouette score ranges from -1 to 1:
        - A score close to 1 indicates that the data points are well clustered (intra-cluster distances are smaller).
        - A score near 0 indicates overlapping clusters.
        - A score close to -1 indicates poor clustering (data points are closer to points in other clusters).
    Args:
        normalized_data (np.ndarray): Normalized motion parameters with shape (n_samples, n_features).
    Returns:
        tuple: 
            - clustering_labels (np.ndarray): Cluster labels for each data point based on the best clustering.
            - cluster_means (np.ndarray): Mean values for each cluster.
            - cluster_stds (np.ndarray): Standard deviations for each cluster.
    """
    best_score = -1  # Initialize the best silhouette score
    best_labels = None  # Store the labels corresponding to the best clustering
    best_means = None  # Mean values for each cluster in the best configuration
    best_stds = None  # Standard deviation values for each cluster in the best configuration
    for n_clusters in range(2, 6):  # Try clustering with 2 to 5 clusters
        print(f"Clustering with {n_clusters} clusters...")
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_data)
        # Compute the silhouette score
        # Silhouette score compares intra-cluster and inter-cluster distances
        score = silhouette_score(normalized_data, cluster_labels)
        print(f"Number of clusters: {n_clusters}, Silhouette score: {score:.3f}")
        # Update the best clustering if the current score is higher
        if score > best_score:
            best_score = score
            best_labels = cluster_labels
            best_means = np.array([
                np.mean(normalized_data[best_labels == i], axis=0) for i in range(n_clusters)
            ])
            best_stds = np.array([
                np.std(normalized_data[best_labels == i], axis=0) for i in range(n_clusters)
            ])
    print(f"Optimal number of clusters determined: {len(best_means)}")
    print(f"Best silhouette score: {best_score:.3f}")
    return best_labels, best_means, best_stds



def _determine_fixation_clusters(cluster_means, cluster_stds):
    """
    Identifies the fixation-related clusters based on cluster means and standard deviations.
    Args:
        cluster_means (np.ndarray): Mean values of the clusters, shape (n_clusters, n_features).
        cluster_stds (np.ndarray): Standard deviation values of the clusters, shape (n_clusters, n_features).
    Returns:
        tuple: 
            fixation_cluster (int): The primary fixation cluster index.
            additional_fixation_clusters (np.ndarray): Indices of additional fixation-related clusters.
    """
    # Determine the primary fixation cluster as the one with the smallest sum of means in the x and y dimensions
    fixation_cluster = np.argmin(np.sum(cluster_means[:, :2], axis=1))
    # Identify additional fixation-related clusters
    additional_fixation_clusters = np.where(
        cluster_means[:, 0] < cluster_means[fixation_cluster, 0] + 3 * cluster_stds[fixation_cluster, 0]
    )[0]
    # Exclude the primary fixation cluster from the additional clusters
    additional_fixation_clusters = additional_fixation_clusters[additional_fixation_clusters != fixation_cluster]
    # Log the identified clusters
    print(f"Primary fixation cluster: {fixation_cluster}")
    print(f"Additional fixation clusters: {additional_fixation_clusters}")
    return fixation_cluster, additional_fixation_clusters



def _classify_clusters_as_fixations_or_non_fixations(clustering_labels, fixation_cluster, additional_fixation_clusters):
    """
    Classifies clusters into fixation-related and non-fixation categories.
    Args:
        clustering_labels (np.ndarray): Array of cluster labels assigned to each point.
        fixation_cluster (int): Primary fixation cluster index.
        additional_fixation_clusters (np.ndarray): Indices of additional fixation-related clusters.
    Returns:
        np.ndarray: Updated labels where:
            1 indicates fixation-related clusters,
            2 indicates non-fixation clusters.
    """
    # Update labels for fixation-related clusters
    updated_labels = np.copy(clustering_labels)
    updated_labels[updated_labels == fixation_cluster] = 100  # Temporary marker for fixations
    for cluster in additional_fixation_clusters:
        updated_labels[updated_labels == cluster] = 100
    # Assign final labels: 1 for fixation-related clusters, 2 for non-fixation clusters
    updated_labels[updated_labels != 100] = 2  # Non-fixation clusters
    updated_labels[updated_labels == 100] = 1  # Fixation-related clusters
    # Log summary of classifications
    print(f"Total fixation-related points: {(updated_labels == 1).sum()}")
    print(f"Total non-fixation points: {(updated_labels == 2).sum()}")
    return updated_labels



def _find_fixation_start_stop_indices(fixation_labels):
    """
    Finds continuous chunks of fixation labels and returns their start and stop indices.
    Args:
        fixation_labels (np.ndarray): Array of fixation labels (1 for fixation, 2 for non-fixation).
    Returns:
        np.ndarray: A 2D array where each row contains [start_index, stop_index] of a fixation.
    """
    # Log the operation for debugging
    # Identify indices where fixation labels are equal to 1
    fixation_indices = np.where(fixation_labels == 1)[0]
    # If no fixations are found, return an empty array
    if len(fixation_indices) == 0:
        print("No fixations found in the labels.")
        return np.empty((0, 2), dtype=int)
    # Find the boundaries of continuous chunks
    chunk_boundaries = np.diff(fixation_indices) > 1
    # Start indices of each fixation chunk
    start_indices = fixation_indices[np.insert(chunk_boundaries, 0, True)]
    # Stop indices of each fixation chunk
    stop_indices = fixation_indices[np.append(chunk_boundaries, True)]
    # Combine start and stop indices into a 2D array
    fixation_chunks = np.column_stack((start_indices, stop_indices))
    # Log the results for debugging
    print(f"Found {len(fixation_chunks)} fixation chunks after global clustering.")
    return fixation_chunks
