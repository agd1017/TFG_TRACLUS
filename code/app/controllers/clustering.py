from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering, Birch
from concurrent.futures import ThreadPoolExecutor
import threading

from models.TRACLUS import traclus as tr
from models.mapping import get_coordinates, map_ilustration, map_heat, plot_map_traclus, plot_clusters_on_map, plot_segments_on_map
from models.data_processing import load_and_simplify_data, relational_table, get_cluster_graph

def get_cluster_trajectories(trajectories, directional=True, use_segments=True, clustering_algorithm=None, 
                            optics_metric=None, optics_algorithm=None, optics_eps=None, optics_sample=None, 
                            dbscan_metric=None, dbscan_algorithm=None, dbscan_eps=None, dbscan_sample=None, 
                            hdbscan_metric=None, hdbscan_algorithm=None, hdbscan_sample=None, 
                            aggl_metric=None, aggl_linkage=None, aggl_n_clusters=None, 
                            spect_affinity=None, spect_assign_labels=None, spect_n_clusters=None):
    
    result = tr(trajectories=trajectories, directional=directional, use_segments=use_segments, clustering_algorithm=clustering_algorithm, 
                optics_min_samples=optics_sample, optics_max_eps=optics_eps, optics_metric=optics_metric, optics_algorithm=optics_algorithm, 
                dbscan_min_samples=dbscan_sample, dbscan_eps=dbscan_eps, dbscan_metric=dbscan_metric, dbscan_algorithm=dbscan_algorithm, 
                hdbscan_min_samples=hdbscan_sample, hdbscan_metric=hdbscan_metric, hdbscan_algorithm=hdbscan_algorithm, 
                spect_n_clusters=spect_n_clusters, spect_affinity=spect_affinity, spect_assign_labels=spect_assign_labels,
                aggl_n_clusters=aggl_n_clusters, aggl_linkage=aggl_linkage, aggl_metric=aggl_metric)
    
    _, segments, _, clusters, cluster_assignments, representative_trajectories = result
    # Representacion de las trayectorias pero sin el primer elemento, este parece ser solo un conjunto basura
    representative_clusters = representative_trajectories[1:]

    return segments, clusters, cluster_assignments, representative_clusters

def get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters):

    traclus_map = plot_map_traclus(representative_clusters)
    
    traclus_map_cluster = plot_clusters_on_map(clusters)
    
    traclus_map_segments = plot_segments_on_map(segments, cluster_assignments)
    
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

    return traclus_map, traclus_map_cluster, traclus_map_segments, tabla_relacional, filtered_cluster_graph

def run_optics(tray, results, lock, optics_metric, optics_algorithm, optics_eps, optics_sample):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=OPTICS, 
            optics_metric=optics_metric, optics_algorithm=optics_algorithm,
            optics_eps=optics_eps, optics_sample=optics_sample
        )
        with lock:
            results['optics'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo optics: {e}')

def run_hdbscan(tray, results, lock, hdbscan_metric, hdbscan_algorithm, hdbscan_sample):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=HDBSCAN,
            hdbscan_metric=hdbscan_metric, hdbscan_algorithm=hdbscan_algorithm,
            hdbscan_sample=hdbscan_sample
        )
        with lock:
            results['hdbscan'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo hdbscan: {e}')

def run_dbscan(tray, results, lock, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=DBSCAN,
            dbscan_metric=dbscan_metric, dbscan_algorithm=dbscan_algorithm,
            dbscan_eps=dbscan_eps, dbscan_sample=dbscan_sample
        )
        with lock:
            results['dbscan'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo dbscan: {e}')

def run_spectral(tray, results, lock, spect_affinity, spect_assign_labels, spect_n_clusters):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=SpectralClustering,
            spect_affinity=spect_affinity, spect_assign_labels=spect_assign_labels,
            spect_n_clusters=spect_n_clusters
        )
        with lock:
            results['spectral'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo spectral: {e}')

def run_agglomerative(tray, results, lock, aggl_metric, aggl_linkage, aggl_n_clusters):
    try:
        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=AgglomerativeClustering,
            aggl_metric=aggl_metric, aggl_linkage=aggl_linkage, aggl_n_clusters=aggl_n_clusters
        )
        with lock:
            results['agglomerative'] = result
    except Exception as e:
        with lock:
            results['errors'].append(f'Error en el algoritmo agglomerative: {e}')

def data_constructor(data, nrows, optics_on, optics_metric, optics_algorithm, optics_eps, optics_sample, 
                dbscan_on, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample, 
                hdbscan_on, hdbscan_metric, hdbscan_algorithm, hdbscan_sample, 
                aggl_on, aggl_metric, aggl_linkage, aggl_n_clusters, 
                spect_on, spect_affinity, spect_assign_labels, spect_n_clusters):
    # Carga de datos
    gdf, tray, df = load_and_simplify_data(data, nrows) 
    minx, miny, maxx, maxy = get_coordinates(gdf)
    html_map = map_ilustration(gdf, minx, miny, maxx, maxy)
    html_heatmap = map_heat(gdf, minx, miny, maxx, maxy)

    traclus_map_optics = traclus_map_cluster_optics = traclus_map_segments_optics = tabla_optics = graph_optics = None
    traclus_map_hdbscan = traclus_map_cluster_hdbscan = traclus_map_segments_hdbscan = tabla_hdbscan= graph_hdbscan = None
    traclus_map_dbscan = traclus_map_cluster_dbscan = traclus_map_segments_dbscan = tabla_dbscan = graph_dbscan = None
    traclus_map_spect = traclus_map_cluster_spect = traclus_map_segments_spect = tabla_spect = graph_spect = None
    traclus_map_aggl = traclus_map_cluster_aggl = traclus_map_segments_aggl = tabla_aggl = graph_aggl = None

    # Resultados compartidos entre hilos
    results = {'optics': None, 'hdbscan': None, 'dbscan': None, 
                'spectral': None, 'agglomerative': None, 'errors': []}
    lock = threading.Lock()

    # Crear hilos para cada algoritmo
    threads = []
    if optics_on:
        t = threading.Thread(target=run_optics, args=(tray, results, lock, optics_metric, optics_algorithm, optics_eps, optics_sample))
        threads.append(t)
    if hdbscan_on:
        t = threading.Thread(target=run_hdbscan, args=(tray, results, lock, hdbscan_metric, hdbscan_algorithm, hdbscan_sample))
        threads.append(t)
    if dbscan_on:
        t = threading.Thread(target=run_dbscan, args=(tray, results, lock, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample))
        threads.append(t)
    if spect_on:
        t = threading.Thread(target=run_spectral, args=(tray, results, lock, spect_affinity, spect_assign_labels, spect_n_clusters))
        threads.append(t)
    if aggl_on:
        t = threading.Thread(target=run_agglomerative, args=(tray, results, lock, aggl_metric, aggl_linkage, aggl_n_clusters))
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
    if optics_on and results.get('optics', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('optics', (None, None, None, None))
        traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)  
    if hdbscan_on and results.get('hdbscan', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('hdbscan', (None, None, None, None))
        traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)
    if dbscan_on and results.get('dbscan', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('dbscan', (None, None, None, None))
        traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)
    if spect_on and results.get('spectral', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('spectral', (None, None, None, None))
        traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)
    if aggl_on and results.get('agglomerative', None):
        segments, clusters, cluster_assignments, representative_clusters = results.get('agglomerative', (None, None, None, None))
        traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl = get_experiment_results(df, segments, clusters, cluster_assignments, representative_clusters)

    # Retornar resultados
    return  gdf, tray, html_map, html_heatmap, \
            traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics, \
            traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan, \
            traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan, \
            traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect, \
            traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl, \
            error_message