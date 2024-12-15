import pytest
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import base64

from app.models.data_processing import load_and_simplify_data
from app.models.TRACLUS import traclus

from sklearn.cluster import OPTICS, DBSCAN, HDBSCAN, AgglomerativeClustering, SpectralClustering
# from sklearn.metrics import pairwise_distances

def create_test_file(file_type='csv', rows=5):
    """
    Creates a temporary test file with trajectory data for testing.

    Parameters:
    - file_type: str, 'csv' or 'excel' to specify the file format.
    - rows: int, number of rows to include in the test file.

    Returns:
    - str, the filename or base64-encoded string for testing.
    """
    data = {
        "TRIP_ID": [f"trip_{i}" for i in range(rows)],
        "POLYLINE": [
            json.dumps([[i, i * 0.5], [i + 0.5, i + 1]]) for i in range(rows)
        ],
    }
    df = pd.DataFrame(data)

    if file_type == 'csv':
        temp_filename = "test_file.csv"
        df.to_csv(temp_filename, index=False)
        with open(temp_filename, "rb") as f:
            encoded_file = base64.b64encode(f.read()).decode('utf-8')
        os.remove(temp_filename)
        return f"data:application/csv;base64,{encoded_file}"
    else:
        temp_filename = "test_file.xlsx"
        df.to_excel(temp_filename, index=False, engine='openpyxl')
        with open(temp_filename, "rb") as f:
            encoded_file = base64.b64encode(f.read()).decode('utf-8')
        os.remove(temp_filename)
        return f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded_file}"

def test_load_and_simplify_data():
    """
    Tests the load_and_simplify_data function by providing mock data and validating outputs.
    """
    # Create a test file (Excel format)
    test_file = create_test_file(file_type='excel', rows=5)

    # Load the file, ensuring compatibility with Excel engine
    try:
        gdf, trajectories, df = load_and_simplify_data(test_file, rows=5, tolerance=0.001)
    except ValueError as e:
        assert "Excel file format" not in str(e), "Ensure 'openpyxl' engine is specified for Excel compatibility."
        raise

    # Validate GeoDataFrame
    assert isinstance(gdf, gpd.GeoDataFrame), "Output gdf is not a GeoDataFrame."
    assert not gdf.empty, "GeoDataFrame is empty."

    # Validate trajectories
    assert isinstance(trajectories, list), "Trajectories output is not a list."
    assert len(trajectories) == 5, "Number of trajectories does not match input rows."

    # Validate DataFrame
    assert isinstance(df, pd.DataFrame), "Output df is not a DataFrame."
    assert not df.empty, "DataFrame is empty."


def test_traclus_optics():
    """
    Tests the traclus function using the OPTICS clustering algorithm.
    """
    trajectories = [np.array([[i, i * 0.5], [i + 0.5, i + 1]]) for i in range(5)]
    result = traclus(trajectories, clustering_algorithm=OPTICS, 
                    optics_min_samples=2, optics_max_eps=10.0, optics_metric='euclidean', optics_algorithm='auto')

    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = result

    assert isinstance(partitions, list), "Partitions output is not a list."
    assert isinstance(segments, list), "Segments output is not a list."
    assert isinstance(dist_matrix, np.ndarray), "Distance matrix is not a numpy array."
    assert len(clusters) > 0, "No clusters were created."
    assert len(cluster_assignments) == len(segments), "Cluster assignments length mismatch."
    assert isinstance(representative_trajectories, list), "Representative trajectories output is not a list."

def test_traclus_dbscan():
    """
    Tests the traclus function using the DBSCAN clustering algorithm.
    """
    trajectories = [np.array([[i, i * 0.5], [i + 0.5, i + 1]]) for i in range(5)]
    result = traclus(trajectories, clustering_algorithm=DBSCAN, 
                    dbscan_min_samples=2, dbscan_eps=1.0, dbscan_metric='euclidean', dbscan_algorithm='auto')

    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = result

    assert isinstance(partitions, list), "Partitions output is not a list."
    assert isinstance(segments, list), "Segments output is not a list."
    assert isinstance(dist_matrix, np.ndarray), "Distance matrix is not a numpy array."
    assert len(clusters) > 0, "No clusters were created."
    assert len(cluster_assignments) == len(segments), "Cluster assignments length mismatch."
    assert isinstance(representative_trajectories, list), "Representative trajectories output is not a list."

def test_traclus_hdbscan():
    """
    Tests the traclus function using the HDBSCAN clustering algorithm.
    """
    # Define example trajectories as input data
    trajectories = [np.array([[i, i * 0.5], [i + 0.5, i + 1]]) for i in range(10)]

    # Run the traclus algorithm using HDBSCAN as the clustering method
    result = traclus(trajectories, clustering_algorithm=HDBSCAN,
            hdbscan_min_samples=2, hdbscan_metric='euclidean', hdbscan_algorithm='auto', 
    )

    # Unpack the results of the traclus function
    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = result

    # Assertions to verify the output
    assert isinstance(partitions, list), "Partitions output is not a list."
    assert isinstance(segments, list), "Segments output is not a list."
    assert isinstance(dist_matrix, np.ndarray), "Distance matrix is not a numpy array."
    assert len(clusters) > 0, "No clusters were created."
    assert len(cluster_assignments) == len(segments), "Cluster assignments length mismatch."
    assert isinstance(representative_trajectories, list), "Representative trajectories output is not a list."

@pytest.mark.filterwarnings("ignore:.*suspiciously like an uncondensed distance matrix.*")
@pytest.mark.filterwarnings("ignore:.*No representative points found.*")
def test_traclus_agglomerative():
    """
    Tests the traclus function using the Agglomerative Clustering algorithm.
    """
    trajectories = [np.array([[i, i * 0.5], [i + 0.5, i + 1]]) for i in range(5)]
    result = traclus(trajectories, clustering_algorithm=AgglomerativeClustering, 
                    mdl_weights=[2, 1, 1], d_weights=[1, 2, 1],    
                    aggl_n_clusters=2, aggl_linkage='ward', aggl_metric='euclidean')

    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = result

    assert isinstance(partitions, list), "Partitions output is not a list."
    assert isinstance(segments, list), "Segments output is not a list."
    assert isinstance(dist_matrix, np.ndarray), "Distance matrix is not a numpy array."
    assert len(clusters) > 0, "No clusters were created."
    assert len(cluster_assignments) == len(segments), "Cluster assignments length mismatch."
    assert isinstance(representative_trajectories, list), "Representative trajectories output is not a list."

@pytest.mark.filterwarnings("ignore:.*No representative points found.*")
@pytest.mark.filterwarnings("ignore:.*Spectral clustering API has changed.*")
def test_traclus_spectral():
    """
    Tests the traclus function using the Spectral Clustering algorithm.
    """
    trajectories = [np.array([[i, i * 0.5], [i + 0.5, i + 1]]) for i in range(5)]
    result = traclus(trajectories, clustering_algorithm=SpectralClustering, 
                    spect_n_clusters=2, spect_affinity='precomputed', spect_assign_labels='kmeans')

    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = result

    assert isinstance(partitions, list), "Partitions output is not a list."
    assert isinstance(segments, list), "Segments output is not a list."
    assert isinstance(dist_matrix, np.ndarray), "Distance matrix is not a numpy array."
    assert len(clusters) > 0, "No clusters were created."
    assert len(cluster_assignments) == len(segments), "Cluster assignments length mismatch."
    assert isinstance(representative_trajectories, list), "Representative trajectories output is not a list."
