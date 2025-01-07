from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering, Birch
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Manager
import matplotlib
matplotlib.use('Agg')
import time

from models.TRACLUS import traclus as tr
from models.mapping import get_coordinates, map_ilustration, map_heat, plot_map_traclus, plot_clusters_on_map, plot_segments_on_map
from models.data_processing import load_and_simplify_data, get_cluster_graph, get_relational_table

# Function to generate clustered trajectories using specified algorithms and parameters.
def get_cluster_trajectories(trajectories, directional=True, use_segments=True, clustering_algorithm=None, 
                            optics_metric=None, optics_algorithm=None, optics_eps=None, optics_sample=None, 
                            dbscan_metric=None, dbscan_algorithm=None, dbscan_eps=None, dbscan_sample=None, 
                            hdbscan_metric=None, hdbscan_algorithm=None, hdbscan_sample=None, 
                            aggl_metric=None, aggl_linkage=None, aggl_n_clusters=None, 
                            spect_affinity=None, spect_assign_labels=None, spect_n_clusters=None):
    """
    Generate trajectory clusters using the specified algorithm and its parameters.
    """
    # Call the TRACLUS method to perform clustering
    result = tr(trajectories=trajectories, directional=directional, use_segments=use_segments, clustering_algorithm=clustering_algorithm, 
                optics_min_samples=optics_sample, optics_max_eps=optics_eps, optics_metric=optics_metric, optics_algorithm=optics_algorithm, 
                dbscan_min_samples=dbscan_sample, dbscan_eps=dbscan_eps, dbscan_metric=dbscan_metric, dbscan_algorithm=dbscan_algorithm, 
                hdbscan_min_samples=hdbscan_sample, hdbscan_metric=hdbscan_metric, hdbscan_algorithm=hdbscan_algorithm, 
                spect_n_clusters=spect_n_clusters, spect_affinity=spect_affinity, spect_assign_labels=spect_assign_labels,
                aggl_n_clusters=aggl_n_clusters, aggl_linkage=aggl_linkage, aggl_metric=aggl_metric)
    
    # Unpack the results of TRACLUS
    _, segments, _, clusters, cluster_assignments, representative_trajectories = result

    # Remove the first trajectory from the representative list (potentially irrelevant)
    representative_clusters = representative_trajectories[1:]

    return segments, clusters, cluster_assignments, representative_clusters

# Function to process results and generate visualizations and data tables.
def get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters):
    """
    Generate maps and tables for clustering results.
    """
    # Generate maps for representative trajectories, clusters, and segments    
    traclus_map = plot_map_traclus(representative_clusters)
    traclus_map_cluster = plot_clusters_on_map(clusters)
    traclus_map_segments = plot_segments_on_map(segments, cluster_assignments)
    
    # Generate relational table and cluster graph in parallel
    def generate_relational_table():
        return get_relational_table(df, segments, cluster_assignments, representative_clusters)
    
    def generate_cluster_graph():
        return get_cluster_graph(cluster_assignments)
    
    # Execute both tasks using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        future_relational_table = executor.submit(generate_relational_table)
        future_cluster_graph = executor.submit(generate_cluster_graph)
        
        # Collect results
        relational_table = future_relational_table.result()
        filtered_cluster_graph = future_cluster_graph.result()

    return traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph

# Functions to run clustering algorithms in separate threads
def run_optics(df, tray, results, optics_metric, optics_algorithm, optics_eps, optics_sample):
    """
    Run TRACLUS with OPTICS clustering algorithm and store the results.
    """
    try:
        start_optics = time.time()

        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=OPTICS, 
            optics_metric=optics_metric, optics_algorithm=optics_algorithm,
            optics_eps=optics_eps, optics_sample=optics_sample
        )
        results['optics'] = result
        segments, clusters, cluster_assignments, representative_clusters = result
        traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph = get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters)
        results['optics_results'] = (traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph)

        end_optics = time.time()

        print(f"Tiempo de ejecución de OPTICS: {end_optics - start_optics} segundos")
    except Exception as e:
        results['errors'].append(f'Error en el algoritmo optics: {e}')

def run_hdbscan(df, tray, results, hdbscan_metric, hdbscan_algorithm, hdbscan_sample):
    """
    Run TRACLUS with HDBSCAN clustering algorithm and store the results.
    """
    try:
        start_hdbscan = time.time()

        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=HDBSCAN,
            hdbscan_metric=hdbscan_metric, hdbscan_algorithm=hdbscan_algorithm,
            hdbscan_sample=hdbscan_sample
        )
        results['hdbscan'] = result
        segments, clusters, cluster_assignments, representative_clusters = result
        traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph = get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters)
        results['hdbscan_results'] = (traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph)

        end_hdbscan = time.time()

        print(f"Tiempo de ejecución de HDBSCAN: {end_hdbscan - start_hdbscan} segundos")
    except Exception as e:
        results['errors'].append(f'Error en el algoritmo hdbscan: {e}')

def run_dbscan(df, tray, results, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample):
    """
    Run TRACLUS with DBSCAN clustering algorithm and store the results.
    """
    try:
        start_dbscan = time.time()

        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=DBSCAN,
            dbscan_metric=dbscan_metric, dbscan_algorithm=dbscan_algorithm,
            dbscan_eps=dbscan_eps, dbscan_sample=dbscan_sample
        )
        results['dbscan'] = result
        segments, clusters, cluster_assignments, representative_clusters = result
        traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph = get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters)
        results['dbscan_results'] = (traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph)

        end_dbscan = time.time()

        print(f"Tiempo de ejecución de DBSCAN: {end_dbscan - start_dbscan} segundos")
    except Exception as e:
        results['errors'].append(f'Error en el algoritmo dbscan: {e}')

def run_spectral(df, tray, results, spect_affinity, spect_assign_labels, spect_n_clusters):
    """
    Run TRACLUS with Spectral clustering algorithm and store the results.
    """
    try:
        start_spectral = time.time()

        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=SpectralClustering,
            spect_affinity=spect_affinity, spect_assign_labels=spect_assign_labels,
            spect_n_clusters=spect_n_clusters
        )
        results['spectral'] = result
        segments, clusters, cluster_assignments, representative_clusters = result
        traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph = get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters)
        results['spectral_results'] = (traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph)

        end_spectral = time.time()

        print(f"Tiempo de ejecución de Spectral: {end_spectral - start_spectral} segundos")
    except Exception as e:
        results['errors'].append(f'Error en el algoritmo spectral: {e}')

def run_agglomerative(df, tray, results, aggl_metric, aggl_linkage, aggl_n_clusters):
    """
    Run TRACLUS with Agglomerative clustering algorithm and store the results.
    """
    try:
        start_agglomerative = time.time()

        result = get_cluster_trajectories(
            trajectories=tray, clustering_algorithm=AgglomerativeClustering,
            aggl_metric=aggl_metric, aggl_linkage=aggl_linkage, aggl_n_clusters=aggl_n_clusters
        )
        results['agglomerative'] = result
        segments, clusters, cluster_assignments, representative_clusters = result
        traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph = get_experiment_results(df, segments, clusters,  cluster_assignments, representative_clusters)
        results['agglomerative_results'] = (traclus_map, traclus_map_cluster, traclus_map_segments, relational_table, filtered_cluster_graph)

        end_agglomerative = time.time()

        print(f"Tiempo de ejecución de Agglomerative: {end_agglomerative - start_agglomerative} segundos")
    except Exception as e:
        results['errors'].append(f'Error en el algoritmo agglomerative: {e}')

def data_constructor(data, nrows, optics_on, optics_metric, optics_algorithm, optics_eps, optics_sample, 
                dbscan_on, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample, 
                hdbscan_on, hdbscan_metric, hdbscan_algorithm, hdbscan_sample, 
                aggl_on, aggl_metric, aggl_linkage, aggl_n_clusters, 
                spect_on, spect_affinity, spect_assign_labels, spect_n_clusters):
    """
    Process input data and run TRACLUS algorithms in parallel, returning results and visualizations.
    """
    # Load and simplify data
    gdf, tray, df = load_and_simplify_data(data, nrows) 
    minx, miny, maxx, maxy = get_coordinates(gdf)
    html_map = map_ilustration(gdf, minx, miny, maxx, maxy)
    html_heatmap = map_heat(gdf, minx, miny, maxx, maxy)

    traclus_map_optics = traclus_map_cluster_optics = traclus_map_segments_optics = table_optics = graph_optics = None
    traclus_map_hdbscan = traclus_map_cluster_hdbscan = traclus_map_segments_hdbscan = table_hdbscan= graph_hdbscan = None
    traclus_map_dbscan = traclus_map_cluster_dbscan = traclus_map_segments_dbscan = table_dbscan = graph_dbscan = None
    traclus_map_spect = traclus_map_cluster_spect = traclus_map_segments_spect = table_spect = graph_spect = None
    traclus_map_aggl = traclus_map_cluster_aggl = traclus_map_segments_aggl = table_aggl = graph_aggl = None

    # Initialize placeholders for results and error handling
    manager = Manager()
    results = manager.dict({'optics': None, 'hdbscan': None, 'dbscan': None, 
                            'spectral': None, 'agglomerative': None, 'errors': []})
    
    # Create and start threads for each enabled clustering algorithm
    processes = []
    if optics_on:
        p = Process(target=run_optics, args=(df, tray, results, optics_metric, optics_algorithm, optics_eps, optics_sample))
        processes.append(p)
    if hdbscan_on:
        p = Process(target=run_hdbscan, args=(df, tray, results, hdbscan_metric, hdbscan_algorithm, hdbscan_sample))
        processes.append(p)
    if dbscan_on:
        p = Process(target=run_dbscan, args=(df, tray, results, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample))
        processes.append(p)
    if spect_on:
        p = Process(target=run_spectral, args=(df, tray, results, spect_affinity, spect_assign_labels, spect_n_clusters))
        processes.append(p)
    if aggl_on:
        p = Process(target=run_agglomerative, args=(df, tray, results, aggl_metric, aggl_linkage, aggl_n_clusters))
        processes.append(p)
    
    # Start and join all threads
    start_traclus = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    end_traclus = time.time()

    print(f"Tiempo de ejecución de TRACLUS: {end_traclus - start_traclus} segundos")

    # Check for errors in clustering
    error_message = None
    if results['errors']:
        error_message = ' | '.join(results['errors'])

    # Unpack results and generate visualizations for each algorithm
    if optics_on and results.get('optics', None):
        traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, table_optics, graph_optics = results.get('optics_results', (None, None, None, None, None))
    if hdbscan_on and results.get('hdbscan', None):
        traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, table_hdbscan, graph_hdbscan = results.get('hdbscan_results', (None, None, None, None, None))
    if dbscan_on and results.get('dbscan', None):
        traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, table_dbscan, graph_dbscan = results.get('dbscan_results', (None, None, None, None, None))
    if spect_on and results.get('spectral', None):
        traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, table_spect, graph_spect = results.get('spectral_results', (None, None, None, None, None))
    if aggl_on and results.get('agglomerative', None):
        traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, table_aggl, graph_aggl = results.get('agglomerative_results', (None, None, None, None, None))

    # Return all results and visualizations
    return  gdf, tray, html_map, html_heatmap, \
            traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, table_optics, graph_optics, \
            traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, table_hdbscan, graph_hdbscan, \
            traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, table_dbscan, graph_dbscan, \
            traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, table_spect, graph_spect, \
            traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, table_aggl, graph_aggl, \
            error_message