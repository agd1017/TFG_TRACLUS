import warnings
import time
import threading
import numpy as np
from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering

#* Optimization and original distance functions

def d_euclidean(points1, points2):
    """
    Computes the Euclidean distance between two sets of points.

    Args:
        points1 (np.ndarray): Array of coordinates for the first set of points.
        points2 (np.ndarray): Array of coordinates for the second set of points.

    Returns:
        np.ndarray: Euclidean distance between the corresponding points.
    """
    # Vectorized calculation of Euclidean distance
    return np.sqrt(np.sum((points1 - points2) ** 2, axis=-1))


def slope_to_rotation_matrix(slope):
    """
    Constructs a rotation matrix based on a given slope.

    Args:
        slope (float): The slope of a line.

    Returns:
        np.ndarray: A 2x2 rotation matrix for the slope.
    """
    # Create a rotation matrix based on the slope
    return np.array([[1, slope], [-slope, 1]])


def get_point_projection_on_line(point, line):
    """
    Projects a point onto a line using rotation and inverse rotation.

    Args:
        point (np.ndarray): The coordinates of the point as [x, y].
        line (np.ndarray): Array of two points defining the line [[x1, y1], [x2, y2]].

    Returns:
        np.ndarray: The coordinates of the projected point on the line.
    """
    # Compute the slope of the line
    line_slope = (line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if line[-1, 0] != line[0, 0] else np.inf

    # Handle vertical lines as a special case
    if np.isinf(line_slope):
        return np.array([line[0, 0], point[1]])

    # Rotate the line and the point
    r = slope_to_rotation_matrix(line_slope)
    rot_line = np.matmul(line, r.T)
    rot_point = np.matmul(point, r.T)

    # Project the point by aligning it with the rotated line
    proj = np.array([rot_point[0], rot_line[0, 1]])

    # Reverse the rotation to obtain the projection in the original space
    r_inverse = np.linalg.inv(r)
    proj = np.matmul(proj, r_inverse.T)

    return proj


def get_point_projection_on_line_optimize(point, line):
    """
    Projects a point onto a line using a vectorized approach.

    Args:
        point (np.ndarray): The coordinates of the point as [x, y].
        line (np.ndarray): Array of two points defining the line [[x1, y1], [x2, y2]].

    Returns:
        np.ndarray: The coordinates of the projected point on the line.
    """
    a = np.array(line[0])  # First point of the line
    b = np.array(line[1])  # Second point of the line
    p = np.array(point)    # The point to project

    # Vectors representing the line (AB) and from the line to the point (AP)
    ab = b - a
    ap = p - a

    # Projection factor using dot product
    dot_product = np.dot(ap, ab)
    ab_square = np.dot(ab, ab)
    factor = dot_product / ab_square

    # Compute the projection as a linear combination
    projection = a + factor * ab

    return projection


def d_perpendicular(l1, l2):
    """
    Calculates the perpendicular distance between two line segments.

    Args:
        l1 (np.ndarray): Array of two points defining the first line segment.
        l2 (np.ndarray): Array of two points defining the second line segment.

    Returns:
        float: The perpendicular distance between the two line segments.
    """
    # Determine which line is shorter
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter, l_longer = l1, l2
    else:
        l_shorter, l_longer = l2, l1

    # Project endpoints of the shorter line onto the longer line
    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    # Compute Lehmer distances
    lehmer_1 = d_euclidean(l_shorter[0], ps)
    lehmer_2 = d_euclidean(l_shorter[-1], pe)

    # Avoid division by zero
    if lehmer_1 == 0 and lehmer_2 == 0:
        return 0
    return (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)


def d_perpendicular_vectorized(vectors, line_lengths):
    """
    Computes the pairwise perpendicular distances between a set of line segments.

    Args:
        vectors (list): A list of line segments, each represented as a numpy array [[x1, y1], [x2, y2]].
        line_lengths (list): A list of lengths for the corresponding line segments.

    Returns:
        np.ndarray: A matrix where the entry (i, j) represents the perpendicular distance between line i and line j.
    """
    n = len(vectors)
    dist_matrix = np.zeros((n, n))  # Initialize the distance matrix

    # Compute distances for each pair of line segments
    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = vectors[i], vectors[j]
            l1_len, l2_len = line_lengths[i], line_lengths[j]

            # Determine which line is shorter
            if l1_len < l2_len:
                l_shorter, l_longer = l1, l2
            else:
                l_shorter, l_longer = l2, l1

            # Project endpoints of the shorter line onto the longer line
            ps = get_point_projection_on_line_optimize(l_shorter[0], l_longer)
            pe = get_point_projection_on_line_optimize(l_shorter[-1], l_longer)

            # Compute Lehmer distances
            lehmer_1 = d_euclidean(l_shorter[0], ps)
            lehmer_2 = d_euclidean(l_shorter[-1], pe)

            # Avoid division by zero
            if lehmer_1 == 0 and lehmer_2 == 0:
                dist_matrix[i, j] = 0
            else:
                dist_matrix[i, j] = (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)

            # Ensure symmetry in the distance matrix
            dist_matrix[j, i] = dist_matrix[i, j]

    return dist_matrix

def d_angular(l1, l2, directional=True):
    """
    Calculate the angular distance between two line segments.
    The angular distance reflects the minimum angle of intersection between the segments,
    optionally scaled by the directional length of the longer line.

    Parameters:
    l1, l2 : ndarray
        The input line segments, each defined by two points in 2D space (start and end).
    directional : bool, optional (default=True)
        If True, scales the angular distance by the length of the longer line segment.

    Returns:
    float
        The angular distance between the two line segments.
    """
    
    # Determine the shorter and longer line segments based on their lengths
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    # Calculate slopes of both lines, handling vertical lines (infinite slope)
    shorter_slope = (l_shorter[-1,1] - l_shorter[0,1]) / (l_shorter[-1,0] - l_shorter[0,0]) if l_shorter[-1,0] - l_shorter[0,0] != 0 else np.inf
    longer_slope = (l_longer[-1,1] - l_longer[0,1]) / (l_longer[-1,0] - l_longer[0,0]) if l_longer[-1,0] - l_longer[0,0] != 0 else np.inf

    # Determine the angle between the two line segments
    theta = None
    if np.isinf(shorter_slope):
        # If the shorter line is vertical, calculate angle with the longer line
        tan_theta0 = longer_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    elif np.isinf(longer_slope):
        # If the longer line is vertical, calculate angle with the shorter line
        tan_theta0 = shorter_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    else:
        # For non-vertical lines, calculate the angle between them using tangent formula
        tan_theta0 = (shorter_slope - longer_slope) / (1 + shorter_slope * longer_slope)
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)

    # If directional scaling is enabled, return the angular distance scaled by the longer line length
    if directional:
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])

    # If not directional, return a scaled angular distance based on the sine of the angle
    if 0 <= theta < (90 * np.pi / 180):
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
    elif (90 * np.pi / 180) <= theta <= np.pi:
        return np.sin(theta)
    else:
        raise ValueError("Theta is not in the range of 0 to 180 degrees.")


def d_angular_vectorized(vectors, line_lengths, directional=True):
    """
    Vectorized calculation of the angular distance matrix for multiple line segments.
    Each pair of line segments is compared to compute their angular distance.

    Parameters:
    vectors : list of ndarray
        List of line segments, each defined by two points in 2D space.
    line_lengths : list of float
        List of line segment lengths, corresponding to the vectors.
    directional : bool, optional (default=True)
        If True, scales the angular distance by the length of the longer line segment.

    Returns:
    ndarray
        A matrix of angular distances between each pair of line segments.
    """
    
    n = len(vectors)
    dist_matrix = np.zeros((n, n))

    # Loop through each pair of line segments to calculate angular distances
    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = vectors[i], vectors[j]
            l_shorter = l_longer = None
            l1_len, l2_len = line_lengths[i], line_lengths[j]
            if l1_len < l2_len:
                l_shorter, l_longer = l1, l2
            else:
                l_shorter, l_longer = l2, l1

            # Calculate the angular distance for this pair of line segments
            shorter_slope = (l_shorter[-1,1] - l_shorter[0,1]) / (l_shorter[-1,0] - l_shorter[0,0]) if l_shorter[-1,0] - l_shorter[0,0] != 0 else np.inf
            longer_slope = (l_longer[-1,1] - l_longer[0,1]) / (l_longer[-1,0] - l_longer[0,0]) if l_longer[-1,0] - l_longer[0,0] != 0 else np.inf

            # Calculate the minimum angle between the two line segments
            theta = None
            if np.isinf(shorter_slope):
                tan_theta0 = longer_slope
                tan_theta1 = tan_theta0 * -1
                theta0 = np.abs(np.arctan(tan_theta0))
                theta1 = np.abs(np.arctan(tan_theta1))
                theta = min(theta0, theta1)
            elif np.isinf(longer_slope):
                tan_theta0 = shorter_slope
                tan_theta1 = tan_theta0 * -1
                theta0 = np.abs(np.arctan(tan_theta0))
                theta1 = np.abs(np.arctan(tan_theta1))
                theta = min(theta0, theta1)
            else:
                tan_theta0 = (shorter_slope - longer_slope) / (1 + shorter_slope * longer_slope)
                tan_theta1 = tan_theta0 * -1
                theta0 = np.abs(np.arctan(tan_theta0))
                theta1 = np.abs(np.arctan(tan_theta1))
                theta = min(theta0, theta1)

            # If directional scaling is enabled, calculate the angular distance with scaling
            if directional:
                dist_matrix[i, j] = np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
            else:
                if 0 <= theta < (90 * np.pi / 180):
                    dist_matrix[i, j] = np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
                elif (90 * np.pi / 180) <= theta <= np.pi:
                    dist_matrix[i, j] = np.sin(theta)
                else:
                    raise ValueError("Theta is not in the range of 0 to 180 degrees.")
            
            # Mirror the distance for symmetry (since distance is symmetric)
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix

def d_parallel_vectorized(vectors, line_lengths):
    """
    Compute the distance matrix based on the parallel relationship between line segments.
    The distance reflects how closely parallel the line segments are when projected onto each other.

    Parameters:
    vectors : list of ndarray
        List of line segments, each defined by two points in 2D space.
    line_lengths : list of float
        List of line segment lengths, corresponding to the vectors.

    Returns:
    ndarray
        A matrix of parallel distances between each pair of line segments.
    """
    
    n = len(vectors)
    dist_matrix = np.zeros((n, n))

    # Loop through each pair of line segments to calculate parallel distances
    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = vectors[i], vectors[j]
            l_shorter = l_longer = None
            l1_len, l2_len = line_lengths[i], line_lengths[j]
            if l1_len < l2_len:
                l_shorter, l_longer = l1, l2
            else:
                l_shorter, l_longer = l2, l1

            # Project the endpoints of the shorter line onto the longer line
            ps = get_point_projection_on_line_optimize(l_shorter[0], l_longer)
            pe = get_point_projection_on_line_optimize(l_shorter[-1], l_longer)

            # Compute the perpendicular distances for both projections
            parallel_1 = min(d_euclidean(l_longer[0], ps), d_euclidean(l_longer[-1], ps))
            parallel_2 = min(d_euclidean(l_longer[0], pe), d_euclidean(l_longer[-1], pe))

            # The parallel distance is the minimum of the two projections
            dist_matrix[i, j] = min(parallel_1, parallel_2)

            # Mirror the distance for symmetry
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def distance_vector(partitions, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):
    """
    Compute a weighted distance vector using parallel, perpendicular, and angular distances between line segments.
    The computation is performed in parallel threads to improve efficiency.

    Parameters:
    partitions : list of ndarray
        List of line segments, each defined by two points in 2D space.
    directional : bool, optional (default=True)
        If True, scales the angular distance by the length of the longer line segment.
    w_perpendicular : float, optional (default=1)
        Weight for the perpendicular distance calculation.
    w_parallel : float, optional (default=1)
        Weight for the parallel distance calculation.
    w_angular : float, optional (default=1)
        Weight for the angular distance calculation.

    Returns:
    float
        The total distance computed as the sum of perpendicular, parallel, and angular distances, each weighted.
    """

    # Containers to store results of each distance type
    results = {
        'perpendicular': None,
        'parallel': None,
        'angular': None
    }

    # Calculate the length of each line segment in the partitions
    line_lengths = [d_euclidean(vec[0], vec[-1]) for vec in partitions]    

    # Define functions that will be executed in separate threads for each type of distance
    def run_perpendicular():
        start_p = time.time()
        results['perpendicular'] = w_perpendicular * d_perpendicular_vectorized(partitions, line_lengths)
        end_p = time.time()
        print("Perpendicular time: ", end_p - start_p)

    def run_parallel():
        start_pa = time.time()
        results['parallel'] = w_parallel * d_parallel_vectorized(partitions, line_lengths)
        end_pa = time.time()
        print("Parallel time: ", end_pa - start_pa)

    def run_angular():
        start_a = time.time()
        results['angular'] = w_angular * d_angular_vectorized(partitions, line_lengths, directional)
        end_a = time.time()
        print("Angular time: ", end_a - start_a)

    # Create threads for each type of distance computation
    thread_perpendicular = threading.Thread(target=run_perpendicular)
    thread_parallel = threading.Thread(target=run_parallel)
    thread_angular = threading.Thread(target=run_angular)

    # Start all threads
    thread_perpendicular.start()
    thread_parallel.start()
    thread_angular.start()

    # Wait for all threads to finish
    thread_perpendicular.join()
    thread_parallel.join()
    thread_angular.join()

    # Sum the results of all distance types
    return results['perpendicular'] + results['parallel'] + results['angular']

def get_vectorice_distance_matrix(partitions, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):
    """
    Compute the distance matrix for a set of line segments, using weighted perpendicular, parallel, and angular distances.
    The distance matrix is computed by calling `distance_vector`.

    Parameters:
    partitions : list of ndarray
        List of line segments, each defined by two points in 2D space.
    directional : bool, optional (default=True)
        If True, scales the angular distance by the length of the longer line segment.
    w_perpendicular : float, optional (default=1)
        Weight for the perpendicular distance calculation.
    w_parallel : float, optional (default=1)
        Weight for the parallel distance calculation.
    w_angular : float, optional (default=1)
        Weight for the angular distance calculation.

    Returns:
    ndarray
        The distance matrix where each element represents the weighted distance between a pair of line segments.
    """

    partitions = np.asarray(partitions)  # Ensure that partitions is a NumPy array
    
    # Get the distance vector, which computes the total distance matrix
    dist_matrix = distance_vector(partitions, directional, w_perpendicular, w_parallel, w_angular)

    # Set diagonal to 0, as the distance from a segment to itself is 0
    np.fill_diagonal(dist_matrix, 0)

    # Check if any NaN values exist in the distance matrix and replace them with a large value
    if np.isnan(dist_matrix).any():
        warnings.warn("The distance matrix contains NaN values")
        np.nan_to_num(dist_matrix, copy=False, nan=9999999)

    return dist_matrix

#* Partitions

def partition2segments(partition):
    """
    Converts a partition of trajectory points into a list of line segments.

    Args:
        partition (np.ndarray): A numpy array of shape (n, 2) representing points in the partition.

    Returns:
        list: A list of numpy arrays, where each array represents a line segment as [[x1, y1], [x2, y2]].
    """
    if not isinstance(partition, np.ndarray):
        raise TypeError("partition must be of type np.ndarray")
    elif partition.shape[1] != 2:
        raise ValueError("partition must be of shape (n, 2)")
    
    segments = []

    # Generate segments by connecting consecutive points
    for i in range(partition.shape[0] - 1):
        segments.append(np.array([[partition[i, 0], partition[i, 1]], 
                                [partition[i + 1, 0], partition[i + 1, 1]]]))
    
    return segments


def minimum_description_length(start_idx, curr_idx, trajectory, w_angular=1, w_perpendicular=1, par=True, directional=True):
    """
    Calculates the minimum description length for a trajectory segment.

    Args:
        start_idx (int): Starting index of the trajectory segment.
        curr_idx (int): Current index being evaluated.
        trajectory (np.ndarray): The full trajectory as a numpy array of shape (n, 2).
        w_angular (float): Weight for angular cost in the MDL calculation.
        w_perpendicular (float): Weight for perpendicular cost in the MDL calculation.
        par (bool): Whether to include partitioning costs in the MDL calculation.
        directional (bool): Whether angular costs should consider directionality.

    Returns:
        float: The minimum description length for the trajectory segment.
    """
    lh = 0  # Length Hypothesis
    ldh = 0  # Length Data Hypothesis

    # Calculate length hypothesis (LH) and data hypothesis (LDH) for the segment
    for i in range(start_idx, curr_idx - 1):
        ed = d_euclidean(trajectory[i], trajectory[i + 1])  # Euclidean distance between consecutive points
        lh += max(0, np.log2(ed, where=ed > 0))  # Avoid log of zero or negative values
        
        if par:
            for j in range(start_idx, i - 1):
                # Calculate perpendicular and angular costs
                ldh += w_perpendicular * d_perpendicular(np.array([trajectory[start_idx], trajectory[i]]), 
                                                        np.array([trajectory[j], trajectory[j + 1]]))
                ldh += w_angular * d_angular(np.array([trajectory[start_idx], trajectory[i]]), 
                                            np.array([trajectory[j], trajectory[j + 1]]), 
                                            directional=directional)

    # Return total cost: LH + LDH (if par is True) or just LH
    if par:
        return lh + ldh
    return lh


def partition(trajectory, directional=True, progress_bar=False, w_perpendicular=1, w_angular=1):
    """
    Partitions a trajectory into segments based on the minimum description length principle.

    Args:
        trajectory (np.ndarray): A numpy array of shape (n, 2) representing the trajectory.
        directional (bool): Whether angular costs should consider directionality.
        progress_bar (bool): Whether to display a progress bar during the partitioning process.
        w_perpendicular (float): Weight for perpendicular cost in the MDL calculation.
        w_angular (float): Weight for angular cost in the MDL calculation.

    Returns:
        np.ndarray: A numpy array containing the control points (start and end points of segments).
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("The trajectory must be a NumPy array.")
    elif trajectory.shape[1] != 2:
        raise ValueError("The trajectory must have shape (n, 2).")
    
    cp_indices = [0]  # List to store control point indices (starts with the first point)
    traj_len = trajectory.shape[0]  # Total number of points in the trajectory
    start_idx = 0  # Start index for the current segment
    length = 1  # Initial segment length

    # Iterate through the trajectory points
    while start_idx + length < traj_len: 
        if progress_bar:
            print(f'\r{round(((start_idx + length) / traj_len) * 100, 2)}%', end='')
        
        curr_idx = start_idx + length  # Current index being evaluated

        # Calculate MDL costs with and without partitioning
        cost_par = minimum_description_length(start_idx, curr_idx, trajectory, 
                                            w_angular=w_angular, w_perpendicular=w_perpendicular, 
                                            directional=directional)
        cost_nopar = minimum_description_length(start_idx, curr_idx, trajectory, 
                                                par=False, directional=directional)

        # If partitioning cost is higher, add a new control point
        if cost_par > cost_nopar: 
            cp_indices.append(curr_idx - 1)  # Add the last valid index as a control point
            start_idx = curr_idx - 1  # Update the start index to the new segment
            length = 1  # Reset segment length
        else:
            length += 1  # Extend the segment

    # Add the last point of the trajectory as a control point
    cp_indices.append(len(trajectory) - 1)

    # Return the trajectory points corresponding to the control points
    return np.array([trajectory[i] for i in cp_indices])

#* Trayectory representation

def get_average_direction_slope(line_list):
    """
    Computes the average slope of a list of line segments.

    Args:
        line_list (list): A list of numpy arrays, where each array represents a line segment 
                            defined by its points (n, 2).

    Returns:
        float: The average slope of the line segments.
    """
    slopes = []

    # Calculate the slope of each line in the list
    for line in line_list:
        # Avoid division by zero when calculating slope
        slopes.append((line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) 
                        if (line[-1, 0] - line[0, 0]) != 0 else 0)

    # Convert the slopes list to a numpy array for easier computation
    slopes = np.array(slopes)

    # Return the mean of all slopes
    return np.mean(slopes)


def get_representative_trajectory(lines, min_lines=3):
    """
    Computes a representative trajectory for a given set of lines.

    Args:
        lines (list): A list of numpy arrays, where each array represents a line segment (n, 2).
        min_lines (int): Minimum number of lines required to determine a representative point.

    Returns:
        numpy.ndarray: An array of representative points forming the trajectory, or an empty array 
                        if no representative trajectory could be computed.
    """
    # Calculate the average slope of the lines
    average_slope = get_average_direction_slope(lines)

    # Create a rotation matrix to align lines based on the average slope
    rotation_matrix = slope_to_rotation_matrix(average_slope)

    # Rotate all lines to align them along a common axis
    rotated_lines = []
    for line in lines:
        rotated_lines.append(np.matmul(line, rotation_matrix.T))

    # Collect all starting and ending points of the rotated lines
    starting_and_ending_points = []
    for line in rotated_lines:
        starting_and_ending_points.append(line[0])  # Starting point
        starting_and_ending_points.append(line[-1])  # Ending point
    starting_and_ending_points = np.array(starting_and_ending_points)

    # Sort points by the x-coordinate to simplify further calculations
    starting_and_ending_points = starting_and_ending_points[starting_and_ending_points[:, 0].argsort()]

    # Initialize a list for representative points
    representative_points = []

    # Iterate over the sorted starting and ending points
    for p in starting_and_ending_points:
        num_p = 0  # Count of lines contributing to the representative point

        # Count how many lines pass through the x-coordinate of point `p`
        for line in rotated_lines:
            point_sorted_line = line[line[:, 0].argsort()]  # Sort line by x-coordinate
            if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                num_p += 1

        # If enough lines contribute, calculate the average y-coordinate
        if num_p >= min_lines:
            y_avg = 0
            for line in rotated_lines:
                point_sorted_line = line[line[:, 0].argsort()]
                if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                    # Add the average of the y-coordinates at the line boundaries
                    y_avg += (point_sorted_line[0, 1] + point_sorted_line[-1, 1]) / 2
            y_avg /= num_p
            representative_points.append(np.array([p[0], y_avg]))

    # If no representative points were found, issue a warning and return an empty array
    if len(representative_points) == 0:
        warnings.warn("WARNING: No representative points found.")
        return np.array([])

    # Convert representative points to a numpy array
    representative_points = np.array(representative_points)

    # Rotate the representative points back to the original coordinate system
    representative_points = np.matmul(representative_points, np.linalg.inv(rotation_matrix).T)

    return representative_points

#* Clustering de las trayectorias con los diversoso algoritmos

def clustering(segments, dist_matrix, clustering_algorithm, 
                optics_min_samples, optics_max_eps, optics_metric, optics_algorithm, 
                dbscan_min_samples, dbscan_eps, dbscan_metric, dbscan_algorithm, 
                hdbscan_min_samples, hdbscan_metric, hdbscan_algorithm, 
                spect_n_clusters, spect_affinity, spect_assign_labels,
                aggl_n_clusters, aggl_linkage, aggl_metric):
    """
    Performs clustering on trajectory segments using various algorithms.

    Args:
        segments (list): List of trajectory segments.
        dist_matrix (numpy.ndarray): Precomputed distance matrix for clustering.
        clustering_algorithm (str): Algorithm to use for clustering.
        optics_min_samples (int): Minimum samples for OPTICS.
        optics_max_eps (float): Maximum epsilon for OPTICS.
        optics_metric (str): Metric for OPTICS.
        optics_algorithm (str): Algorithm used by OPTICS.
        dbscan_min_samples (int): Minimum samples for DBSCAN.
        dbscan_eps (float): Epsilon for DBSCAN.
        dbscan_metric (str): Metric for DBSCAN.
        dbscan_algorithm (str): Algorithm used by DBSCAN.
        hdbscan_min_samples (int): Minimum samples for HDBSCAN.
        hdbscan_metric (str): Metric for HDBSCAN.
        hdbscan_algorithm (str): Algorithm used by HDBSCAN.
        spect_n_clusters (int): Number of clusters for Spectral Clustering.
        spect_affinity (str): Affinity for Spectral Clustering.
        spect_assign_labels (str): Label assignment method for Spectral Clustering.
        aggl_n_clusters (int): Number of clusters for Agglomerative Clustering.
        aggl_linkage (str): Linkage method for Agglomerative Clustering.
        aggl_metric (str): Metric for Agglomerative Clustering.

    Returns:
        tuple: A tuple containing the list of clusters and cluster assignments.
    """
    clusters = []
    clustering_model = None

    # Select clustering algorithm and initialize with parameters
    if clustering_algorithm == OPTICS:
        params = {'min_samples': optics_min_samples, 'max_eps': optics_max_eps, 
                'metric': optics_metric, 'algorithm': optics_algorithm}
        clustering_model = OPTICS(**params)

    elif clustering_algorithm == DBSCAN:
        params = {'min_samples': dbscan_min_samples, 'eps': dbscan_eps, 
                'metric': dbscan_metric, 'algorithm': dbscan_algorithm}
        clustering_model = DBSCAN(**params)

    elif clustering_algorithm == HDBSCAN:
        params = {'min_samples': hdbscan_min_samples, 'metric': hdbscan_metric, 
                'algorithm': hdbscan_algorithm}
        clustering_model = HDBSCAN(**params)

    elif clustering_algorithm == SpectralClustering:
        params = {'n_clusters': spect_n_clusters, 'affinity': spect_affinity, 
                'assign_labels': spect_assign_labels}
        clustering_model = SpectralClustering(**params)

    elif clustering_algorithm == AgglomerativeClustering:
        params = {'n_clusters': aggl_n_clusters, 'linkage': aggl_linkage, 
                'metric': aggl_metric}
        clustering_model = AgglomerativeClustering(**params)

    # Ensure a clustering model was created
    if clustering_model is not None:
        cluster_assignments = clustering_model.fit_predict(dist_matrix)
        # Group segments into clusters based on their assignments
        clusters = [[segments[i] for i in np.nonzero(cluster_assignments == c)[0]] for c in np.unique(cluster_assignments)]
    else:
        raise ValueError("Invalid clustering algorithm or parameters.")

    return clusters, cluster_assignments

#* Main function

def traclus(trajectories, directional=True, use_segments=True, clustering_algorithm=OPTICS, 
            mdl_weights=[1, 1, 1], d_weights=[1, 1, 1], 
            optics_min_samples=None, optics_max_eps=None, optics_metric=None, optics_algorithm=None, 
            dbscan_min_samples=None, dbscan_eps=None, dbscan_metric=None, dbscan_algorithm=None, 
            hdbscan_min_samples=None, hdbscan_metric=None, hdbscan_algorithm=None, 
            spect_n_clusters=None, spect_affinity=None, spect_assign_labels=None,
            aggl_n_clusters=None, aggl_linkage=None, aggl_metric=None):
    """
    Implements the TRACLUS trajectory clustering algorithm.

    Args:
        trajectories (list): List of trajectories, where each trajectory is a numpy array of shape (n, 2).
        directional (bool): If True, considers directionality in partitioning and distance calculations.
        use_segments (bool): If True, uses segments for clustering; otherwise, uses partitions.
        clustering_algorithm (str): Algorithm to use for clustering.
        mdl_weights (list): Weights for perpendicular, parallel, and angular MDL costs.
        d_weights (list): Weights for perpendicular, parallel, and angular distances.
        (Additional args): Parameters for specific clustering algorithms (e.g., OPTICS, DBSCAN, etc.).

    Returns:
        tuple: A tuple containing partitions, segments, distance matrix, clusters, cluster assignments, 
                and representative trajectories.
    """
    # Validate trajectories format
    if not isinstance(trajectories, list):
        raise TypeError("Trajectories must be a list.")
    for trajectory in trajectories:
        if not isinstance(trajectory, np.ndarray):
            raise TypeError("Each trajectory must be a numpy array.")
        elif len(trajectory.shape) != 2 or trajectory.shape[1] != 2:
            raise ValueError("Each trajectory must be a numpy array of shape (n, 2).")

    # Partition trajectories using MDL-based partitioning
    start_partitions = time.time()
    partitions = []
    for trajectory in trajectories:
        partitions.append(partition(trajectory, directional=directional, progress_bar=False, 
                                    w_perpendicular=mdl_weights[0], w_angular=mdl_weights[2]))
    end_partitions = time.time()
    print("Partitioning time: ", end_partitions - start_partitions)

    # Convert partitions to segments if specified
    start_segments = time.time()
    segments = []
    if use_segments:
        for parts in partitions:
            segments += partition2segments(parts)
    else:
        segments = partitions
    end_segments = time.time()
    print("Segments time: ", end_segments - start_segments)

    # Compute the distance matrix for clustering
    start_vectorization = time.time()
    dist_matrix = get_vectorice_distance_matrix(segments, directional=directional, 
                                                w_perpendicular=d_weights[0], 
                                                w_parallel=d_weights[1], 
                                                w_angular=d_weights[2])
    end_vectorization = time.time()
    print("Vectorization time: ", end_vectorization - start_vectorization)

    # Perform clustering
    start_clustering = time.time()
    clusters, cluster_assignments = clustering(segments, dist_matrix, clustering_algorithm, 
                                                optics_min_samples, optics_max_eps, optics_metric, optics_algorithm, 
                                                dbscan_min_samples, dbscan_eps, dbscan_metric, dbscan_algorithm, 
                                                hdbscan_min_samples, hdbscan_metric, hdbscan_algorithm, 
                                                spect_n_clusters, spect_affinity, spect_assign_labels,
                                                aggl_n_clusters, aggl_linkage, aggl_metric)
    end_clustering = time.time()
    print("Clustering time: ", end_clustering - start_clustering)

    # Extract representative trajectories for each cluster
    start_representatives = time.time()
    representative_trajectories = []
    for cluster in clusters:
        representative_trajectories.append(get_representative_trajectory(cluster))
    end_representatives = time.time()
    print("Representatives time: ", end_representatives - start_representatives)

    return partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories
