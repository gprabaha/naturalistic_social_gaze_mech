
import numpy as np
import time
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score

import pdb

def detect_fixation_in_position_array(positions, session_name, samprate=1/1000):
    fix_params = _get_fixation_parameters(session_name, samprate)
    if positions.shape[0] > int(30 / (fix_params['samprate'] * 1000)):    
        print("\nPreprocessing positions data for fixation detection")
        x, y = _preprocess_data(positions, fix_params)
        print("Extracting vel, accel, etc. parameters for k-means clustering")
        dist, vel, accel, rot = _extract_motion_parameters(x, y)
        print("Normalizing parameters for k-means clustering")
        normalized_data_params = _normalize_motion_parameters(dist, vel, accel, rot)
        print("Performing global clustering of points for 2 to 5 cluster size")
        clustering_labels, cluster_means, cluster_stds = _kmeans_cluster_all_points_globally(normalized_data_params[:,1:]) # exclude distance for clustering
        print("Determining fixation cluster based on smallest mean velocity and additional clusters with velocity within 3sd velocity of fixation cluster")
        fixation_cluster, additional_fixation_clusters = _determine_fixation_related_clusters(cluster_means, cluster_stds)
        print("Updating point labels to fixation and not-fixation clusters based on previous analysis")
        fixation_labels = _classify_clusters_as_fixations_or_non_fixations(
            clustering_labels, fixation_cluster, additional_fixation_clusters)
        print("Calculating the start and stop indices of each fixation")
        fixation_start_stop_indices = _find_fixation_start_stop_indices(fixation_labels)
        print("Refining fixation start-stop indices using local reclustering")
        refined_fixation_start_stop_indices = _refine_fixation_start_stop_with_reclustering(
            fixation_start_stop_indices, normalized_data_params, padding=50, fix_params=fix_params)
        return np.reshape(refined_fixation_start_stop_indices, (-1, 2)) \
            if refined_fixation_start_stop_indices.size > 0 else np.empty((0, 2), dtype=int)
    else:
        print("\n!! Data too short for fixation detection processing !!\n")
        return np.empty((0, 2), dtype=int) 



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
        sample_size_for_silhouette_scoring = min(100000, len(normalized_data))  # Subsample if data size exceeds 10,000
        score = silhouette_score(normalized_data, cluster_labels, sample_size=sample_size_for_silhouette_scoring, random_state=42)
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



def _determine_fixation_related_clusters(cluster_means, cluster_stds):
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



def _refine_fixation_start_stop_with_reclustering(fixation_start_stop_indices, normalized_data_params, padding=50, fix_params=None):
    """
    Refines fixation start-stop indices by performing local reclustering within each fixation window.
    Fixations are split into smaller valid chunks if non-fixation points are detected.
    Args:
        fixation_start_stop_indices (np.ndarray): 2D array of fixation start and stop indices.
        normalized_data_params (np.ndarray): Normalized feature parameters (e.g., velocity, acceleration).
        padding (int): Number of points to pad before and after each fixation for reclustering.
        fix_params (dict): Fixation parameters dictionary containing sampling rate (samprate).
    Returns:
        np.ndarray: Updated 2D array of refined fixation start and stop indices.
    """
    refined_fixation_indices = []
    for start, stop in fixation_start_stop_indices:
        # Extract fixation window and extend with padding for reclustering
        extended_indices = np.arange(start - padding, stop + padding + 1)
        extended_indices = extended_indices[(extended_indices >= 0) & (extended_indices < len(normalized_data_params))]
        data_subset = normalized_data_params[extended_indices]
        # Determine the optimal number of clusters using silhouette scores
        max_clusters = min(6, len(data_subset))
        if max_clusters <= 1:
            refined_fixation_indices.append([start, stop])
            continue
        silhouette_scores = []
        for num_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, n_init=5, random_state=42)
            labels = kmeans.fit_predict(data_subset)
            score = silhouette_score(data_subset, labels)
            silhouette_scores.append((num_clusters, score))
        if not silhouette_scores:
            refined_fixation_indices.append([start, stop])
            continue
        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        # Perform KMeans clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(data_subset)
        # Identify the fixation cluster as the one with the smallest median velocity and acceleration
        median_values = np.array([
            np.median(data_subset[cluster_labels == i], axis=0) for i in range(optimal_clusters)
        ])
        fixation_cluster = np.argmin(np.sum(median_values[:, [1, 2]], axis=1))  # Velocity at index 1, acceleration at index 2
        # Include additional fixation clusters based on velocity and acceleration thresholds
        additional_fixation_clusters = np.where(
            (median_values[:, 1] < median_values[fixation_cluster, 1] + 3 * np.std(data_subset[cluster_labels == fixation_cluster][:, 1])) &
            (median_values[:, 2] < median_values[fixation_cluster, 2] + 3 * np.std(data_subset[cluster_labels == fixation_cluster][:, 2]))
        )[0]
        additional_fixation_clusters = additional_fixation_clusters[additional_fixation_clusters != fixation_cluster]
        # Mark all fixation clusters
        fixation_mask = np.isin(cluster_labels, np.concatenate(([fixation_cluster], additional_fixation_clusters)))
        # Extract non-fixation points based on the original fixation window
        original_indices = np.arange(start, stop + 1)
        relative_indices_within_original = np.isin(extended_indices, original_indices)
        non_fixation_relative_indices = np.where(
            ~fixation_mask[relative_indices_within_original]
        )[0]
        # Split the original fixation start-stop indices based on non-fixation points
        if len(non_fixation_relative_indices) == 0:
            refined_fixation_indices.append([start, stop])
        else:
            valid_chunks = __split_fixations_at_non_fixations(
                start, stop, non_fixation_relative_indices + start
            )
            refined_fixation_indices.extend(valid_chunks)
    # Post-process refined fixations
    refined_fixation_indices = np.array(refined_fixation_indices)
    if refined_fixation_indices.size > 0:
        # Check for invalid start-stop intervals and warn if found
        invalid_indices = refined_fixation_indices[refined_fixation_indices[:, 0] >= refined_fixation_indices[:, 1]]
        if invalid_indices.size > 0:
            print(f"\nWARNING: Detected {len(invalid_indices)} invalid fixation intervals where stop index is before or equal to start index.")
            print(f"Invalid intervals: {invalid_indices}\n")
        # Remove invalid intervals
        refined_fixation_indices = refined_fixation_indices[refined_fixation_indices[:, 0] < refined_fixation_indices[:, 1]]
        # Apply minimum duration threshold
        min_duration = int(0.025 / fix_params['samprate'])
        refined_fixation_indices = refined_fixation_indices[
            (refined_fixation_indices[:, 1] - refined_fixation_indices[:, 0] + 1) >= min_duration
        ]
    return refined_fixation_indices

def __split_fixations_at_non_fixations(start, stop, non_fix_indices):
    """
    Splits a fixation into valid periods by excluding non-fixation points.
    Args:
        start (int): Start index of the fixation.
        stop (int): Stop index of the fixation.
        non_fix_indices (np.ndarray): Indices of non-fixation points within the fixation.
    Returns:
        list: List of valid fixation start and stop indices.
    """
    valid_chunks = []
    current_start = start
    for non_fix in non_fix_indices:
        if current_start < non_fix:
            valid_chunks.append([current_start, non_fix - 1])
        current_start = non_fix + 1
    if current_start <= stop:
        valid_chunks.append([current_start, stop])
    return valid_chunks

