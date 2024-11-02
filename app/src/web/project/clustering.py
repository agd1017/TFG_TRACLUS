from TRACLUS import traclus as tr
from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering, Birch
from data_processing import load_and_simplify_data, relational_table
from mapping import map_ilustration, map_heat, plot_map_traclus, plot_map_traclus_df, get_coordinates

def get_cluster_trajectories(df, trajectories, directional=True, use_segments=True, clustering_algorithm=None, 
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
    
    _, segments, _, _, cluster_assignments, representative_trajectories = result
    # Representacion de las trayectorias pero sin el primer elemento, este parece ser solo un conjunto basura
    representative_clusters = representative_trajectories[1:representative_trajectories.__len__()]

    TRACLUS_map = plot_map_traclus(representative_clusters)
    TRACLUS_map_df = plot_map_traclus_df(representative_clusters, df['POLYLINE'])
    tabla_relacional = relational_table(df, segments, cluster_assignments)

    return TRACLUS_map, TRACLUS_map_df, tabla_relacional

def constructor(data, nrows, OPTICS_ON, OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample, 
                DBSCAN_ON, DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample, 
                HDBSCAN_ON, HDBSCAN_metric, HDBSCAN_algorithm, HDBSCAN_sample, 
                Aggl_ON, Aggl_metric, Aggl_linkage, Aggl_n_clusters, 
                Spect_ON, Spect_affinity, Spect_assign_labels, Spect_n_clusters):
    # Carga de datos
    gdf, tray, df = load_and_simplify_data(data, nrows) 
    minx, miny, maxx, maxy = get_coordinates(gdf)
    html_map = map_ilustration(gdf, minx, miny, maxx, maxy)
    html_heatmap = map_heat(gdf, minx, miny, maxx, maxy)

    # Inicializar variables para los resultados
    TRACLUS_map_OPTICS = TRACLUS_map_df_OPTICS = tabla_OPTICS = None
    TRACLUS_map_HDBSCAN = TRACLUS_map_df_HDBSCAN = tabla_HDBSCAN = None
    TRACLUS_map_DBSCAN = TRACLUS_map_df_DBSCAN = tabla_DBSCAN = None
    TRACLUS_map_Spect = TRACLUS_map_df_Spect = tabla_Spect = None
    TRACLUS_map_Aggl = TRACLUS_map_df_Aggl = tabla_Aggl = None
    error_messages = []  # Inicializar lista de mensajes de error

    # Algoritmo OPTICS
    if OPTICS_ON:
        try:
            TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS, tabla_OPTICS = get_cluster_trajectories(
                df=df, trajectories=tray, clustering_algorithm=OPTICS, 
                OPTICS_metric=OPTICS_metric, OPTICS_algorithm=OPTICS_algorithm,
                OPTICS_eps=OPTICS_eps, OPTICS_sample=OPTICS_sample
            )
        except Exception as e:
            error_messages.append(f'Error en el algoritmo OPTICS: {e}')

    # Algoritmo HDBSCAN
    if HDBSCAN_ON:
        try:
            TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, tabla_HDBSCAN = get_cluster_trajectories(
                df=df, trajectories=tray, clustering_algorithm=HDBSCAN,
                HDBSCAN_metric=HDBSCAN_metric, HDBSCAN_algorithm=HDBSCAN_algorithm,
                HDBSCAN_sample=HDBSCAN_sample
            )
        except Exception as e:
            error_messages.append(f'Error en el algoritmo HDBSCAN: {e}')

    # Algoritmo DBSCAN
    if DBSCAN_ON:
        try:
            TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, tabla_DBSCAN = get_cluster_trajectories(
                df=df, trajectories=tray, clustering_algorithm=DBSCAN,
                DBSCAN_metric=DBSCAN_metric, DBSCAN_algorithm=DBSCAN_algorithm,
                DBSCAN_eps=DBSCAN_eps, DBSCAN_sample=DBSCAN_sample
            )
        except Exception as e:
            error_messages.append(f'Error en el algoritmo DBSCAN: {e}')

    # Algoritmo SpectralClustering
    if Spect_ON:
        try:
            TRACLUS_map_Spect, TRACLUS_map_df_Spect, tabla_Spect = get_cluster_trajectories(
                df=df, trajectories=tray, clustering_algorithm=SpectralClustering, 
                Spect_assign_labels=Spect_assign_labels, Spect_n_clusters=Spect_n_clusters,
                Spect_affinity=Spect_affinity
            )
        except Exception as e:
            error_messages.append(f'Error en el algoritmo SpectralClustering: {e}')

    # Algoritmo AgglomerativeClustering
    if Aggl_ON:
        try:
            TRACLUS_map_Aggl, TRACLUS_map_df_Aggl, tabla_Aggl = get_cluster_trajectories(
                df=df, trajectories=tray, clustering_algorithm=AgglomerativeClustering, 
                Aggl_metric=Aggl_metric, Aggl_n_clusters=Aggl_n_clusters, 
                Aggl_linkage=Aggl_linkage
            )
        except Exception as e:
            error_messages.append(f'Error en el algoritmo AgglomerativeClustering: {e}')

    # Verificar si hubo errores
    error_message = None
    if error_messages:
        error_message = ' | '.join(error_messages)  # Concatenar todos los mensajes de error

    return gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
            TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, \
            TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_Spect, \
            TRACLUS_map_df_Spect, TRACLUS_map_Aggl, TRACLUS_map_df_Aggl, \
            tabla_OPTICS, tabla_HDBSCAN, tabla_DBSCAN, tabla_Spect, tabla_Aggl, error_message