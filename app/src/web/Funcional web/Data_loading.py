from Funtions import *
from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering, Birch
# Constructor

def constructor(data, nrows, OPTICS_ON, OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample, 
                DBSCAN_ON, DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample, 
                HDBSCAN_ON, HDBSCAN_metric, HDBSCAN_algorithm, HDBSCAN_sample, 
                Aggl_ON, Aggl_metric, Aggl_linkage, Aggl_n_clusters, 
                Spect_ON, Spect_affinity, Spect_assign_labels, Spect_n_clusters):
    # Carga de datos
    gdf, tray, df = load_and_simplify_data(data, nrows) 
    minx, miny, maxx, maxy = solicitar_coordenadas(gdf)
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