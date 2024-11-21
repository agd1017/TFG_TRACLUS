from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering, Birch
from concurrent.futures import ThreadPoolExecutor
import threading

from models.TRACLUS import traclus as tr
from models.mapping import get_coordinates, map_ilustration, map_heat, plot_map_traclus, plot_clusters_on_map, plot_segments_on_map
from models.data_processing import load_and_simplify_data, relational_table, get_cluster_graph

def get_cluster_trajectories(trajectories, directional=True, use_segments=True, clustering_algorithm=None, 
                            OPTICS_metric=None, OPTICS_algorithm=None, OPTICS_eps=None, OPTICS_sample=None, 
                            DBSCAN_metric=None, DBSCAN_algorithm=None, DBSCAN_eps=None, DBSCAN_sample=None, 
                            HDBSCAN_metric=None, HDBSCAN_algorithm=None, HDBSCAN_sample=None, 
                            Aggl_metric=None, Aggl_linkage=None, Aggl_n_clusters=None, 
                            Spect_affinity=None, Spect_assign_labels=None, Spect_n_clusters=None):
    
    result = tr(trajectories=trajectories, directional=directional, use_segments=use_segments, clustering_algorithm=clustering_algorithm, 
                OPTICS_min_samples=OPTICS_sample, OPTICS_max_eps=OPTICS_eps, OPTICS_metric=OPTICS_metric, OPTICS_algorithm=OPTICS_algorithm, 
                DBSCAN_min_samples=DBSCAN_sample, DBSCAN_eps=DBSCAN_eps, DBSCAN_metric=DBSCAN_metric, DBSCAN_algorithm=DBSCAN_algorithm, 
                HDBSCAN_min_samples=HDBSCAN_sample, HDBSCAN_metric=HDBSCAN_metric, HDBSCAN_algorithm=HDBSCAN_algorithm, 
                Spect_n_clusters=Spect_n_clusters, Spect_affinity=Spect_affinity, Spect_assign_labels=Spect_assign_labels,
                Aggl_n_clusters=Aggl_n_clusters, Aggl_linkage=Aggl_linkage, Aggl_metric=Aggl_metric)
    
    _, segments, _, clusters, cluster_assignments, representative_trajectories = result
    # Representacion de las trayectorias pero sin el primer elemento, este parece ser solo un conjunto basura
    representative_clusters = representative_trajectories[1:]

    return segments, clusters, cluster_assignments, representative_clusters

def get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters):

    TRACLUS_map = plot_map_traclus(representative_clusters)
    
    TRACLUS_map_cluster = plot_clusters_on_map(clusters)
    
    TRACLUS_map_segments = plot_segments_on_map(segments, cluster_assignments)
    
    def generate_relational_table():
        return relational_table(df, segments, cluster_assignments, representative_clusters)
    
    def generate_cluster_graph():
        return get_cluster_graph(cluster_assignments)
    
    # Ejecutar las funciones en paralelo
    with ThreadPoolExecutor() as executor:
        future_relational_table = executor.submit(generate_relational_table)
        future_cluster_graph = executor.submit(generate_cluster_graph)
        
        # Obtener resultados
        tabla_relacional = future_relational_table.result()
        filtered_cluster_graph = future_cluster_graph.result()

    return TRACLUS_map, TRACLUS_map_cluster, TRACLUS_map_segments, tabla_relacional, filtered_cluster_graph

def run_optics(tray, results, lock, OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=OPTICS, 
            OPTICS_metric=OPTICS_metric, OPTICS_algorithm=OPTICS_algorithm,
            OPTICS_eps=OPTICS_eps, OPTICS_sample=OPTICS_sample
        )
        with lock:
            results['OPTICS'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo OPTICS: {e}')

def run_hdbscan(tray, results, lock, HDBSCAN_metric, HDBSCAN_algorithm, HDBSCAN_sample):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=HDBSCAN,
            HDBSCAN_metric=HDBSCAN_metric, HDBSCAN_algorithm=HDBSCAN_algorithm,
            HDBSCAN_sample=HDBSCAN_sample
        )
        with lock:
            results['HDBSCAN'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo HDBSCAN: {e}')

def run_dbscan(tray, results, lock, DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=DBSCAN,
            DBSCAN_metric=DBSCAN_metric, DBSCAN_algorithm=DBSCAN_algorithm,
            DBSCAN_eps=DBSCAN_eps, DBSCAN_sample=DBSCAN_sample
        )
        with lock:
            results['DBSCAN'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo DBSCAN: {e}')

def run_spectral(tray, results, lock, Spect_affinity, Spect_assign_labels, Spect_n_clusters):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=SpectralClustering,
            Spect_affinity=Spect_affinity, Spect_assign_labels=Spect_assign_labels,
            Spect_n_clusters=Spect_n_clusters
        )
        with lock:
            results['Spectral'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo Spectral: {e}')

def run_agglomerative(tray, results, lock, Aggl_metric, Aggl_linkage, Aggl_n_clusters):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=AgglomerativeClustering,
            Aggl_metric=Aggl_metric, Aggl_linkage=Aggl_linkage, Aggl_n_clusters=Aggl_n_clusters
        )
        with lock:
            results['Agglomerative'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo Agglomerative: {e}')

def data_constructor(data, nrows, OPTICS_ON, OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample, 
                DBSCAN_ON, DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample, 
                HDBSCAN_ON, HDBSCAN_metric, HDBSCAN_algorithm, HDBSCAN_sample, 
                Aggl_ON, Aggl_metric, Aggl_linkage, Aggl_n_clusters, 
                Spect_ON, Spect_affinity, Spect_assign_labels, Spect_n_clusters):
    # Carga de datos
    gdf, tray, df = load_and_simplify_data(data, nrows) 
    minx, miny, maxx, maxy = get_coordinates(gdf)
    html_map = map_ilustration(gdf, minx, miny, maxx, maxy)
    html_heatmap = map_heat(gdf, minx, miny, maxx, maxy)

    TRACLUS_map_OPTICS = TRACLUS_map_cluster_OPTICS = TRACLUS_map_segments_OPTICS = tabla_OPTICS = graph_OPTICS = None
    TRACLUS_map_HDBSCAN = TRACLUS_map_cluster_HDBSCAN = TRACLUS_map_segments_HDBSCAN = tabla_HDBSCAN= graph_HDBSCAN = None
    TRACLUS_map_DBSCAN = TRACLUS_map_cluster_DBSCAN = TRACLUS_map_segments_DBSCAN = tabla_DBSCAN = graph_DBSCAN = None
    TRACLUS_map_Spect = TRACLUS_map_cluster_Spect = TRACLUS_map_segments_Spect = tabla_Spect = graph_Spect = None
    TRACLUS_map_Aggl = TRACLUS_map_cluster_Aggl = TRACLUS_map_segments_Aggl = tabla_Aggl = graph_Aggl = None

    # Resultados compartidos entre hilos
    results = {'OPTICS': None, 'HDBSCAN': None, 'DBSCAN': None, 
                'Spectral': None, 'Agglomerative': None, 'errors': []}
    lock = threading.Lock()

    # Crear hilos para cada algoritmo
    threads = []
    if OPTICS_ON:
        t = threading.Thread(target=run_optics, args=(tray, results, lock, OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample))
        threads.append(t)
    if HDBSCAN_ON:
        t = threading.Thread(target=run_hdbscan, args=(tray, results, lock, HDBSCAN_metric, HDBSCAN_algorithm, HDBSCAN_sample))
        threads.append(t)
    if DBSCAN_ON:
        t = threading.Thread(target=run_dbscan, args=(tray, results, lock, DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample))
        threads.append(t)
    if Spect_ON:
        t = threading.Thread(target=run_spectral, args=(tray, results, lock, Spect_affinity, Spect_assign_labels, Spect_n_clusters))
        threads.append(t)
    if Aggl_ON:
        t = threading.Thread(target=run_agglomerative, args=(tray, results, lock, Aggl_metric, Aggl_linkage, Aggl_n_clusters))
        threads.append(t)
    
    # Iniciar todos los hilos
    for t in threads:
        t.start()

    # Esperar a que todos los hilos terminen
    for t in threads:
        t.join()

    # Verificar errores
    error_message = None
    if results['errors']:
        error_message = ' | '.join(results['errors'])

    # Desempaquetar los resultados para devolverlos como antes
    if OPTICS_ON and results.get('OPTICS', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('OPTICS', (None, None, None, None))
        TRACLUS_map_OPTICS, TRACLUS_map_cluster_OPTICS, TRACLUS_map_segments_OPTICS, tabla_OPTICS, graph_OPTICS = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)  
    if HDBSCAN_ON and results.get('HDBSCAN', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('HDBSCAN', (None, None, None, None))
        TRACLUS_map_HDBSCAN, TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_segments_HDBSCAN, tabla_HDBSCAN, graph_HDBSCAN = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)
    if DBSCAN_ON and results.get('DBSCAN', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('DBSCAN', (None, None, None, None))
        TRACLUS_map_DBSCAN, TRACLUS_map_cluster_DBSCAN, TRACLUS_map_segments_DBSCAN, tabla_DBSCAN, graph_DBSCAN = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)
    if Spect_ON and results.get('Spectral', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('Spectral', (None, None, None, None))
        TRACLUS_map_Spect, TRACLUS_map_cluster_Spect, TRACLUS_map_segments_Spect, tabla_Spect, graph_Spect = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)
    if Aggl_ON and results.get('Agglomerative', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('Agglomerative', (None, None, None, None))
        TRACLUS_map_Aggl, TRACLUS_map_cluster_Aggl, TRACLUS_map_segments_Aggl, tabla_Aggl, graph_Aggl = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)

    # Retornar resultados
    return gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
            TRACLUS_map_segments_OPTICS, TRACLUS_map_HDBSCAN,TRACLUS_map_cluster_OPTICS, \
            TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_cluster_DBSCAN, \
            TRACLUS_map_cluster_Spect, TRACLUS_map_cluster_Aggl, TRACLUS_map_segments_HDBSCAN, \
            TRACLUS_map_DBSCAN, TRACLUS_map_segments_DBSCAN, TRACLUS_map_Spect, \
            TRACLUS_map_segments_Spect, TRACLUS_map_Aggl, TRACLUS_map_segments_Aggl, \
            tabla_OPTICS, tabla_HDBSCAN, tabla_DBSCAN, tabla_Spect, tabla_Aggl, \
            graph_OPTICS, graph_HDBSCAN, graph_DBSCAN, graph_Spect, graph_Aggl, error_message