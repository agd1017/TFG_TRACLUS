import numpy as np
from sklearn.cluster import OPTICS
import warnings
import time
import threading
import numpy as np
from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering

#* Optimizacion provisional y forma original

def d_euclidean(points1, points2):
    # Vectorized Euclidean distance
    return np.sqrt(np.sum((points1 - points2) ** 2, axis=-1))

def slope_to_rotation_matrix(slope):
    
    return np.array([[1, slope], [-slope, 1]])

def get_point_projection_on_line(point, line):

    line_slope = (line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if line[-1, 0] != line[0, 0] else np.inf

    if np.isinf(line_slope):
        return np.array([line[0,0], point[1]])
    
    r = slope_to_rotation_matrix(line_slope)

    rot_line = np.matmul(line, r.T)
    rot_point = np.matmul(point, r.T)

    proj = np.array([rot_point[0], rot_line[0,1]])

    r_inverse = np.linalg.inv(r)
    proj = np.matmul(proj, r_inverse.T)

    return proj

def get_point_projection_on_line_optimize(point, line):
    a = np.array(line[0])
    b = np.array(line[1])
    p = np.array(point)
    
    # Vectores AB y AP
    ab = b - a
    ap = p - a
    
    # Producto escalar de AB y AP, y producto escalar de AB con sí mismo
    dot_product = np.dot(ap, ab)
    ab_square = np.dot(ab, ab)
    
    # Factor de proyección sobre AB
    factor = dot_product / ab_square
    
    # Proyección en términos de vector
    projection = a + factor * ab
    
    return projection

def d_perpendicular(l1, l2):

    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    lehmer_1 = d_euclidean(l_shorter[0], ps)
    lehmer_2 = d_euclidean(l_shorter[-1], pe)

    if lehmer_1 == 0 and lehmer_2 == 0:
        return 0
    return (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)

def d_perpendicular_vectorized(vectors, line_lengths):
    n = len(vectors)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = vectors[i], vectors[j]
            l1_len, l2_len = line_lengths[i], line_lengths[j]
            if l1_len < l2_len:
                l_shorter, l_longer = l1, l2
            else:
                l_shorter, l_longer = l2, l1

            ps = get_point_projection_on_line_optimize(l_shorter[0], l_longer)
            pe = get_point_projection_on_line_optimize(l_shorter[-1], l_longer)
            
            lehmer_1 = d_euclidean(l_shorter[0], ps)
            lehmer_2 = d_euclidean(l_shorter[-1], pe)

            if lehmer_1 == 0 and lehmer_2 == 0:
                dist_matrix[i, j] = 0
            else:
                dist_matrix[i, j] = (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)

            dist_matrix[j, i] = dist_matrix[i, j]

    return dist_matrix

def d_angular(l1, l2, directional=True):

    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    shorter_slope = (l_shorter[-1,1] - l_shorter[0,1]) / (l_shorter[-1,0] - l_shorter[0,0]) if l_shorter[-1,0] - l_shorter[0,0] != 0 else np.inf
    longer_slope = (l_longer[-1,1] - l_longer[0,1]) / (l_longer[-1,0] - l_longer[0,0]) if l_longer[-1,0] - l_longer[0,0] != 0 else np.inf

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

    if directional:
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])

    if 0 <= theta < (90 * np.pi / 180):
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
    elif (90 * np.pi / 180) <= theta <= np.pi:
        return np.sin(theta)
    else:
        raise ValueError("Theta is not in the range of 0 to 180 degrees.")

def d_angular_vectorized(vectors, line_lengths,  directional=True):
    n = len(vectors)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = vectors[i], vectors[j]
            l_shorter = l_longer = None
            l1_len, l2_len = line_lengths[i], line_lengths[j]
            if l1_len < l2_len:
                l_shorter, l_longer = l1, l2
            else:
                l_shorter, l_longer = l2, l1

            # Get the minimum intersecting angle between both lines
            shorter_slope = (l_shorter[-1,1] - l_shorter[0,1]) / (l_shorter[-1,0] - l_shorter[0,0]) if l_shorter[-1,0] - l_shorter[0,0] != 0 else np.inf
            longer_slope = (l_longer[-1,1] - l_longer[0,1]) / (l_longer[-1,0] - l_longer[0,0]) if l_longer[-1,0] - l_longer[0,0] != 0 else np.inf

            # The case of a vertical line
            theta = None
            if np.isinf(shorter_slope):
                # Get the angle of the longer line with the x-axis and subtract it from 90 degrees
                tan_theta0 = longer_slope
                tan_theta1 = tan_theta0 * -1
                theta0 = np.abs(np.arctan(tan_theta0))
                theta1 = np.abs(np.arctan(tan_theta1))
                theta = min(theta0, theta1)
            elif np.isinf(longer_slope):
                # Get the angle of the shorter line with the x-axis and subtract it from 90 degrees
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

            if directional:
                dist_matrix[i, j] = np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
            else:
                if 0 <= theta < (90 * np.pi / 180):
                    dist_matrix[i, j] = np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
                elif (90 * np.pi / 180) <= theta <= np.pi:
                    dist_matrix[i, j] = np.sin(theta)
                else:
                    raise ValueError("Theta is not in the range of 0 to 180 degrees.")
            
            dist_matrix[j, i] = dist_matrix[i, j]
            
    return dist_matrix

def d_parallel_vectorized(vectors, line_lengths):
    n = len(vectors)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = vectors[i], vectors[j]
            l_shorter = l_longer = None
            l1_len, l2_len = line_lengths[i], line_lengths[j]
            if l1_len < l2_len:
                l_shorter, l_longer = l1, l2
            else:
                l_shorter, l_longer = l2, l1

            ps = get_point_projection_on_line_optimize(l_shorter[0], l_longer)
            pe = get_point_projection_on_line_optimize(l_shorter[-1], l_longer)

            parallel_1 = min(d_euclidean(l_longer[0], ps), d_euclidean(l_longer[-1], ps))
            parallel_2 = min(d_euclidean(l_longer[0], pe), d_euclidean(l_longer[-1], pe))

            dist_matrix[i, j] = min(parallel_1, parallel_2)

            dist_matrix[j, i] = dist_matrix[i, j]
        
    return dist_matrix

def distance_vector(partitions, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):

    # Contenedores para los resultados
    results = {
        'perpendicular': None,
        'parallel': None,
        'angular': None
    }

    line_lengths = [d_euclidean(vec[0], vec[-1]) for vec in partitions]    

    # Definir las funciones que se ejecutarán en cada thread
    def run_perpendicular():
        inicio_p = time.time()
        results['perpendicular'] = w_perpendicular * d_perpendicular_vectorized(partitions, line_lengths)
        fin_p = time.time()
        print("Tiempo perpendicular: ", fin_p - inicio_p)

    def run_parallel():
        inicio_pa = time.time()
        results['parallel'] = w_parallel * d_parallel_vectorized(partitions, line_lengths)
        fin_pa = time.time()
        print("Tiempo paralelo: ", fin_pa - inicio_pa)

    def run_angular():
        inicio_a = time.time()
        results['angular'] = w_angular * d_angular_vectorized(partitions, line_lengths, directional)
        fin_a = time.time()
        print("Tiempo angular: ", fin_a - inicio_a)

    # Crear los threads
    thread_perpendicular = threading.Thread(target=run_perpendicular)
    thread_parallel = threading.Thread(target=run_parallel)
    thread_angular = threading.Thread(target=run_angular)

    # Iniciar los threads
    thread_perpendicular.start()
    thread_parallel.start()
    thread_angular.start()

    # Esperar a que todos los threads terminen
    thread_perpendicular.join()
    thread_parallel.join()
    thread_angular.join()

    # Sumar los resultados
    return results['perpendicular'] + results['parallel'] + results['angular']

def get_vectorice_distance_matrix(partitions, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):
    partitions = np.asarray(partitions)  # Asegurar que partitions sea un array de NumPy
    n_partitions = len(partitions)
    dist_matrix = np.zeros((n_partitions, n_partitions))
    
    dist_matrix = distance_vector(partitions, directional, w_perpendicular, w_parallel, w_angular)

    np.fill_diagonal(dist_matrix, 0)

    if np.isnan(dist_matrix).any():
        warnings.warn("La matriz de distancias contiene valores NaN")
        np.nan_to_num(dist_matrix, copy=False, nan=9999999)

    return dist_matrix

#* Particiones

def partition2segments(partition):

    if not isinstance(partition, np.ndarray):
        raise TypeError("partition must be of type np.ndarray")
    elif partition.shape[1] != 2:
        raise ValueError("partition must be of shape (n, 2)")
    
    segments = []
    for i in range(partition.shape[0]-1):
        segments.append(np.array([[partition[i, 0], partition[i, 1]], [partition[i+1, 0], partition[i+1, 1]]]))

    return segments

def minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=1, w_perpendicular=1, par=True, directional=True):
    
    LH = LDH = 0
    for i in range(start_idx, curr_idx-1):
        ed = d_euclidean(trajectory[i], trajectory[i+1])
        LH += max(0, np.log2(ed, where=ed>0))
        if par:
            for j in range(start_idx, i-1):
                LDH += w_perpendicular * d_perpendicular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]))
                LDH += w_angular * d_angular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]), directional=directional)
    if par:
        return LH + LDH
    return LH

def partition(trajectory, directional=True, progress_bar=False, w_perpendicular=1, w_angular=1):
    
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("La trayectoria debe ser un arreglo de NumPy")
    elif trajectory.shape[1] != 2:
        raise ValueError("La trayectoria debe tener la forma (n, 2)")
    
    cp_indices = [0]

    traj_len = trajectory.shape[0]
    start_idx = 0  
    
    length = 1  
    while start_idx + length < traj_len: 
        if progress_bar:
            print(f'\r{round(((start_idx + length) / traj_len) * 100, 2)}%', end='')
        
        curr_idx = start_idx + length 

        cost_par = minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=w_angular, w_perpendicular=w_perpendicular, directional=directional)
        cost_nopar = minimum_desription_length(start_idx, curr_idx, trajectory, par=False, directional=directional)

        if cost_par > cost_nopar: 
            cp_indices.append(curr_idx-1)  
            start_idx = curr_idx-1
            length = 1
        else:
            length += 1 

    cp_indices.append(len(trajectory) - 1)

    return np.array([trajectory[i] for i in cp_indices])

#* Representacion de trayecorias

def get_average_direction_slope(line_list):

    slopes = []
    for line in line_list:
        slopes.append((line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if (line[-1, 0] - line[0, 0]) != 0 else 0)
    slopes = np.array(slopes)

    return np.mean(slopes)

def get_representative_trajectory(lines, min_lines=3):
    
    average_slope = get_average_direction_slope(lines)
    rotation_matrix = slope_to_rotation_matrix(average_slope)

    rotated_lines = []
    for line in lines:
        rotated_lines.append(np.matmul(line, rotation_matrix.T))

    starting_and_ending_points = []
    for line in rotated_lines:
        starting_and_ending_points.append(line[0])
        starting_and_ending_points.append(line[-1])
    starting_and_ending_points = np.array(starting_and_ending_points)

    starting_and_ending_points = starting_and_ending_points[starting_and_ending_points[:, 0].argsort()]

    representative_points = []
    for p in starting_and_ending_points:
        num_p = 0
        for line in rotated_lines:
            point_sorted_line = line[line[:, 0].argsort()]
            if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                num_p += 1

        if num_p >= min_lines:
            y_avg = 0
            for line in rotated_lines:
                point_sorted_line = line[line[:, 0].argsort()]
                if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                    y_avg += (point_sorted_line[0, 1] + point_sorted_line[-1, 1]) / 2
            y_avg /= num_p
            representative_points.append(np.array([p[0], y_avg]))

    if len(representative_points) == 0:
        warnings.warn("ADVERTENCIA: No se encontraron puntos representativos.")
        return np.array([])

    representative_points = np.array(representative_points)
    representative_points = np.matmul(representative_points, np.linalg.inv(rotation_matrix).T)

    return representative_points

#* Clustering de las trayectorias con los diversoso algoritmos

def clustering(segments, dist_matrix, clustering_algorithm, 
                optics_min_samples, optics_max_eps, optics_metric, optics_algorithm, 
                dbscan_min_samples, dbscan_eps, dbscan_metric, dbscan_algorithm, 
                hdbscan_min_samples, hdbscan_metric, hdbscan_algorithm, 
                spect_n_clusters, spect_affinity, spect_assign_labels,
                aggl_n_clusters, aggl_linkage, aggl_metric):
    
    clusters = []
    clustering_model = None

    # Para OPTICS
    if clustering_algorithm == OPTICS:
        params = {'min_samples': optics_min_samples, 'max_eps': optics_max_eps, 'metric': optics_metric, 'algorithm': optics_algorithm}
        clustering_model = OPTICS(**params)

    # Para DBSCAN
    elif clustering_algorithm == DBSCAN:
        params = {'min_samples': dbscan_min_samples, 'eps': dbscan_eps, 'metric': dbscan_metric, 'algorithm': dbscan_algorithm}
        clustering_model = DBSCAN(**params)

    # Para HDBSCAN
    elif clustering_algorithm == HDBSCAN:
        params = {'min_samples': hdbscan_min_samples, 'metric': hdbscan_metric, 'algorithm': hdbscan_algorithm}
        clustering_model = HDBSCAN(**params)

    # Para SpectralClustering
    elif clustering_algorithm == SpectralClustering:
        params = {'n_clusters': spect_n_clusters, 'affinity': spect_affinity, 'assign_labels': spect_assign_labels}
        clustering_model = SpectralClustering(**params)

    # Para AgglomerativeClustering
    elif clustering_algorithm == AgglomerativeClustering:
        params = {'n_clusters': aggl_n_clusters, 'linkage': aggl_linkage, 'metric': aggl_metric}
        clustering_model = AgglomerativeClustering(**params)

    # Asegurar que el modelo se ha creado antes de continuar
    if clustering_model is not None:
        cluster_assignments = clustering_model.fit_predict(dist_matrix)
        clusters = [[segments[i] for i in np.where(cluster_assignments == c)[0]] for c in np.unique(cluster_assignments)]
    else:
        raise ValueError("No se proporcionó un algoritmo de clustering válido o no se pudo crear el modelo.")

    return clusters, cluster_assignments

#* Traclus

def traclus(trajectories, directional=True, use_segments=True, clustering_algorithm=OPTICS, mdl_weights=[1,1,1], d_weights=[1,1,1], 
            optics_min_samples=None, optics_max_eps=None, optics_metric=None, optics_algorithm=None, 
            dbscan_min_samples=None, dbscan_eps=None, dbscan_metric=None, dbscan_algorithm=None, 
            hdbscan_min_samples=None, hdbscan_metric=None, hdbscan_algorithm=None, 
            spect_n_clusters=None, spect_affinity=None, spect_assign_labels=None,
            aggl_n_clusters=None, aggl_linkage=None, aggl_metric=None):
    """
        Trajectory Clustering Algorithm
    """
    # Ensure that the trajectories are a list of numpy arrays of shape (n, 2)
    if not isinstance(trajectories, list):
        raise TypeError("Trajectories must be a list")
    for trajectory in trajectories:
        if not isinstance(trajectory, np.ndarray):
            raise TypeError("Trajectories must be a list of numpy arrays")
        elif len(trajectory.shape) != 2:
            raise ValueError("Trajectories must be a list of numpy arrays of shape (n, 2)")
        elif trajectory.shape[1] != 2:
            raise ValueError("Trajectories must be a list of numpy arrays of shape (n, 2)")

    # Partition trajectories
    partitions = []
    for trajectory in trajectories:
        partitions.append(partition(trajectory, directional=directional, progress_bar=False, w_perpendicular=mdl_weights[0], w_angular=mdl_weights[2]))

    # Convert partitions to segments
    segments = []
    if use_segments:
        for parts in partitions:
            segments += partition2segments(parts)
    else:
        segments = partitions

    # Get distance matrix
    dist_matrix = get_vectorice_distance_matrix(segments, directional=directional, w_perpendicular=d_weights[0], w_parallel=d_weights[1], w_angular=d_weights[2])

    # Clustering
    clusters, cluster_assignments = clustering(segments, dist_matrix, clustering_algorithm, 
                                                optics_min_samples, optics_max_eps, optics_metric, optics_algorithm, 
                                                dbscan_min_samples, dbscan_eps, dbscan_metric, dbscan_algorithm, 
                                                hdbscan_min_samples, hdbscan_metric, hdbscan_algorithm, 
                                                spect_n_clusters, spect_affinity, spect_assign_labels,
                                                aggl_n_clusters, aggl_linkage, aggl_metric)

    # Get the representative trajectories
    representative_trajectories = []
    for cluster in clusters:
        representative_trajectories.append(get_representative_trajectory(cluster))

    return partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories