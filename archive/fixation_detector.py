
import numpy as np
import logging
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fixation_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def detect_fixation_in_position_array(positions, session_name, samprate=1/1000):
    fix_params = _get_fixation_parameters(session_name, samprate)
    min_points = int(100 / (fix_params['samprate'] * 1000))
    if positions.shape[0] > min_points:    
        logger.info("Preprocessing positions data for fixation detection")
        x, y = _preprocess_data(positions, fix_params)
        logger.info("Extracting motion parameters for k-means clustering")
        dist, vel, accel, rot = _extract_motion_parameters(x, y)
        logger.info("Normalizing parameters for k-means clustering")
        normalized_data_params = _normalize_motion_parameters(dist, vel, accel, rot)
        logger.info("Performing global clustering of points for 2 to 5 cluster sizes")
        clustering_labels, cluster_means, cluster_stds = _kmeans_cluster_all_points_globally(normalized_data_params[:, 1:])
        logger.info("Determining fixation-related clusters based on velocity metrics")
        fixation_cluster, additional_fixation_clusters = _determine_fixation_related_clusters(cluster_means, cluster_stds)
        logger.info("Classifying clusters into fixation and non-fixation")
        fixation_labels = _classify_clusters_as_fixations_or_non_fixations(
            clustering_labels, fixation_cluster, additional_fixation_clusters)
        logger.info("Calculating fixation start and stop indices")
        fixation_start_stop_indices = _find_fixation_start_stop_indices(fixation_labels)
        logger.info("Refining fixation start-stop indices using local reclustering")
        refined_fixation_start_stop_indices = _refine_fixation_start_stop_with_reclustering(
            fixation_start_stop_indices, normalized_data_params, padding=50, fix_params=fix_params)
        return np.reshape(refined_fixation_start_stop_indices, (-1, 2)) \
            if refined_fixation_start_stop_indices.size > 0 else np.empty((0, 2), dtype=int)
    else:
        logger.warning("Data too short for fixation detection processing")
        return np.empty((0, 2), dtype=int)


def _get_fixation_parameters(session_name=None, samprate=1/1000, num_cpus=1):
    logger.info("Initializing fixation parameters")
    variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
    fltord = 60
    lowpasfrq = 30
    nyqfrq = 1000 / 2
    flt = signal.firwin2(fltord, 
                         [0, lowpasfrq / nyqfrq, lowpasfrq / nyqfrq, 1], 
                         [1, 1, 0, 0])
    buffer = int(100 / (samprate * 1000))
    return {
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


def _preprocess_data(positions, fix_params):
    logger.info("Preprocessing x and y data")
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
    logger.info("Resampling data")
    t_old = np.linspace(0, len(data) - 1, len(data))
    resample_factor = fix_params['samprate'] * 1000
    if resample_factor > 1:
        logger.error("Resample factor is too large: %f", resample_factor)
        raise ValueError("Resample factor is too large, leading to excessive memory usage.")
    t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
    f = interp1d(t_old, data, kind='linear')
    return f(t_new)


def __apply_filter(data, fix_params):
    logger.info("Applying low-pass filter")
    return signal.filtfilt(fix_params['flt'], 1, data)


def _extract_motion_parameters(x, y):
    logger.info("Extracting velocity, acceleration, angle, distance, and rotation parameters")
    velx = np.diff(x)
    vely = np.diff(y)
    accelx = np.diff(velx)
    accely = np.diff(vely)
    vel = np.sqrt(velx ** 2 + vely ** 2)
    accel = np.sqrt(accelx ** 2 + accely ** 2)
    angle = np.degrees(np.arctan2(vely, velx))
    vel = vel[:-1]
    rot = np.zeros(len(x) - 2)
    dist = np.zeros(len(x) - 2)
    for a in range(len(x) - 2):
        rot[a] = np.abs(angle[a] - angle[a + 1])
        dist[a] = np.sqrt((x[a] - x[a + 2]) ** 2 + (y[a] - y[a + 2]) ** 2)
    rot[rot > 180] = 360 - rot[rot > 180]
    return dist, vel, accel, rot


def _normalize_motion_parameters(dist, vel, accel, rot):
    logger.info("Normalizing motion parameters")
    parameters = np.stack([dist, vel, accel, rot], axis=1)
    for i in tqdm(range(parameters.shape[1]), desc="Normalizing parameters"):
        max_val = np.mean(parameters[:, i]) + 3 * np.std(parameters[:, i])
        parameters[:, i] = np.clip(parameters[:, i], None, max_val)
        min_val = np.min(parameters[:, i])
        max_val = np.max(parameters[:, i])
        parameters[:, i] = (parameters[:, i] - min_val) / (max_val - min_val) if max_val > min_val else 0
    return parameters


def _kmeans_cluster_all_points_globally(normalized_data):
    logger.info("Performing KMeans clustering globally")
    best_score = -1
    best_labels = None
    best_means = None
    best_stds = None
    for n_clusters in tqdm(range(2, 6), desc="KMeans Clustering"):
        logger.info("Clustering with %d clusters...", n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_data)
        sample_size = min(100000, len(normalized_data))
        score = silhouette_score(normalized_data, cluster_labels, sample_size=sample_size, random_state=42)
        logger.info("Number of clusters: %d, Silhouette score: %.3f", n_clusters, score)
        if score > best_score:
            best_score = score
            best_labels = cluster_labels
            best_means = np.array([
                np.mean(normalized_data[best_labels == i], axis=0) for i in range(n_clusters)
            ])
            best_stds = np.array([
                np.std(normalized_data[best_labels == i], axis=0) for i in range(n_clusters)
            ])
    logger.info("Optimal number of clusters: %d, Best silhouette score: %.3f", len(best_means), best_score)
    return best_labels, best_means, best_stds


def _determine_fixation_related_clusters(cluster_means, cluster_stds):
    """
    Identifies the fixation-related clusters based on cluster means and standard deviations.
    """
    logger.info("Determining fixation-related clusters")
    fixation_cluster = np.argmin(np.sum(cluster_means[:, :2], axis=1))
    additional_fixation_clusters = np.where(
        cluster_means[:, 0] < cluster_means[fixation_cluster, 0] + 3 * cluster_stds[fixation_cluster, 0]
    )[0]
    additional_fixation_clusters = additional_fixation_clusters[additional_fixation_clusters != fixation_cluster]
    logger.info("Primary fixation cluster: %d", fixation_cluster)
    logger.info("Additional fixation clusters: %s", additional_fixation_clusters)
    return fixation_cluster, additional_fixation_clusters


def _classify_clusters_as_fixations_or_non_fixations(clustering_labels, fixation_cluster, additional_fixation_clusters):
    """
    Classifies clusters into fixation-related and non-fixation categories.
    """
    logger.info("Classifying clusters into fixation and non-fixation")
    updated_labels = np.copy(clustering_labels)
    updated_labels[updated_labels == fixation_cluster] = 100  # Temporary marker for fixations
    for cluster in additional_fixation_clusters:
        updated_labels[updated_labels == cluster] = 100
    updated_labels[updated_labels != 100] = 2  # Non-fixation clusters
    updated_labels[updated_labels == 100] = 1  # Fixation-related clusters
    logger.info("Total fixation-related points: %d", (updated_labels == 1).sum())
    logger.info("Total non-fixation points: %d", (updated_labels == 2).sum())
    return updated_labels


def _find_fixation_start_stop_indices(fixation_labels):
    """
    Finds continuous chunks of fixation labels and returns their start and stop indices.
    """
    logger.info("Finding start and stop indices for fixations")
    fixation_indices = np.where(fixation_labels == 1)[0]
    if len(fixation_indices) == 0:
        logger.warning("No fixations found in the labels.")
        return np.empty((0, 2), dtype=int)
    chunk_boundaries = np.diff(fixation_indices) > 1
    start_indices = fixation_indices[np.insert(chunk_boundaries, 0, True)]
    stop_indices = fixation_indices[np.append(chunk_boundaries, True)]
    fixation_chunks = np.column_stack((start_indices, stop_indices))
    logger.info("Found %d fixation chunks after global clustering", len(fixation_chunks))
    return fixation_chunks


def _refine_fixation_start_stop_with_reclustering(fixation_start_stop_indices, normalized_data_params, padding=50, fix_params=None):
    """
    Refines fixation start-stop indices by performing local reclustering within each fixation window.
    """
    logger.info("Refining fixation start-stop indices with local reclustering")
    refined_fixation_indices = []
    for start, stop in tqdm(fixation_start_stop_indices, desc="Refining fixations"):
        extended_indices = np.arange(start - padding, stop + padding + 1)
        extended_indices = extended_indices[(extended_indices >= 0) & (extended_indices < len(normalized_data_params))]
        data_subset = normalized_data_params[extended_indices]
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
        kmeans = KMeans(n_clusters=optimal_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(data_subset)
        median_values = np.array([
            np.median(data_subset[cluster_labels == i], axis=0) for i in range(optimal_clusters)
        ])
        fixation_cluster = np.argmin(np.sum(median_values[:, [1, 2]], axis=1))
        additional_fixation_clusters = np.where(
            (median_values[:, 1] < median_values[fixation_cluster, 1] + 3 * np.std(data_subset[cluster_labels == fixation_cluster][:, 1])) &
            (median_values[:, 2] < median_values[fixation_cluster, 2] + 3 * np.std(data_subset[cluster_labels == fixation_cluster][:, 2]))
        )[0]
        additional_fixation_clusters = additional_fixation_clusters[additional_fixation_clusters != fixation_cluster]
        fixation_mask = np.isin(cluster_labels, np.concatenate(([fixation_cluster], additional_fixation_clusters)))
        original_indices = np.arange(start, stop + 1)
        relative_indices_within_original = np.isin(extended_indices, original_indices)
        non_fixation_relative_indices = np.where(
            ~fixation_mask[relative_indices_within_original]
        )[0]
        if len(non_fixation_relative_indices) == 0:
            refined_fixation_indices.append([start, stop])
        else:
            valid_chunks = __split_fixations_at_non_fixations(
                start, stop, non_fixation_relative_indices + start
            )
            refined_fixation_indices.extend(valid_chunks)
    refined_fixation_indices = np.array(refined_fixation_indices)
    if refined_fixation_indices.size > 0:
        invalid_indices = refined_fixation_indices[refined_fixation_indices[:, 0] >= refined_fixation_indices[:, 1]]
        if invalid_indices.size > 0:
            logger.warning("Detected %d invalid fixation intervals where stop index is before or equal to start index.", len(invalid_indices))
        refined_fixation_indices = refined_fixation_indices[refined_fixation_indices[:, 0] < refined_fixation_indices[:, 1]]
        min_duration = int(0.05 / fix_params['samprate'])
        refined_fixation_indices = refined_fixation_indices[
            (refined_fixation_indices[:, 1] - refined_fixation_indices[:, 0] + 1) >= min_duration
        ]
    return refined_fixation_indices

def __split_fixations_at_non_fixations(start, stop, non_fix_indices):
    """
    Splits a fixation into valid periods by excluding non-fixation points.
    """
    logger.info("Splitting fixations at non-fixation points")
    valid_chunks = []
    current_start = start
    for non_fix in non_fix_indices:
        if current_start < non_fix:
            valid_chunks.append([current_start, non_fix - 1])
        current_start = non_fix + 1
    if current_start <= stop:
        valid_chunks.append([current_start, stop])
    logger.info("Split into %d valid chunks", len(valid_chunks))
    return valid_chunks

