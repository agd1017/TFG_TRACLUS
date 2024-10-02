from Funtions import *
from sklearn.cluster import OPTICS, HDBSCAN, DBSCAN, SpectralClustering, AgglomerativeClustering, Birch

# Constructor

def constructor (data, nrows):

    # Carga de fichero datos
    gdf, tray, df = load_and_simplify_data(data, nrows)

    # Carga mapa ilustrado (cordenadas provisionales, puede que se añadan carga con difernetes cordenadas a la vez)
    html_map = map_ilustration(gdf, -8.689, 41.107, -8.560, 41.185)

    # Carga mapa de calor
    html_heatmap = map_heat(gdf, -8.689, 41.107, -8.560, 41.185)

    # Carga de representaciones de trayectorias traclus
    # Algortmo OPTICS
    TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS, n_clusters = get_cluster_trajectories(df=df, trajectories=tray, clustering_algorithm=OPTICS)
    print(n_clusters)

    # Algortmo HDBSCAN
    TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, _ = get_cluster_trajectories(df=df, trajectories=tray, clustering_algorithm=HDBSCAN)
    
    # Algortmo DBSCAN
    TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, _ = get_cluster_trajectories(df=df, trajectories=tray, clustering_algorithm=DBSCAN, max_eps=0.1)

    # Algortmo SpectralClustering
    TRACLUS_map_SpectralClustering, TRACLUS_map_df_SpectralClustering, _ = get_cluster_trajectories(df=df, trajectories=tray, clustering_algorithm=SpectralClustering, n_clusters = n_clusters , affinity = 'nearest_neighbors')

    # Algortmo AgglomerativeClustering
    TRACLUS_map_AgglomerativeClustering, TRACLUS_map_df_AgglomerativeClustering, _ = get_cluster_trajectories(df=df, trajectories=tray,clustering_algorithm=AgglomerativeClustering, n_clusters = n_clusters, linkage = 'ward')

    """ # Algortmo Birch
    TRACLUS_map_Birch, TRACLUS_map_df_Birch = get_cluster_trajectories(df=df, trajectories=tray, clustering_algorithm=Birch, n_clusters = n_clusters, threshold=1) """

    return gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_SpectralClustering, TRACLUS_map_df_SpectralClustering, TRACLUS_map_AgglomerativeClustering, TRACLUS_map_df_AgglomerativeClustering


""" data = "C:/Users/Álvaro/Documents/GitHub/TFG/TFG_TRACLUS/app/train_data/taxis_trajectory/train.csv"
nrows = 10
constructor (data, nrows) """