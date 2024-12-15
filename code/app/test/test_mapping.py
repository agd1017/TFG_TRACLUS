import pytest
import geopandas as gpd
from shapely.geometry import LineString
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg') 

from app.models.mapping import map_ilustration, map_heat, plot_map_traclus, plot_clusters_on_map, plot_segments_on_map

@pytest.fixture
def mock_geodataframe():
    """
    Create a mock GeoDataFrame with sample LineString geometries.
    """
    data = [
        LineString([(0, 0), (1, 1), (2, 2)]),
        LineString([(2, 2), (3, 3), (4, 4)]),
        LineString([(4, 4), (5, 5), (6, 6)])
    ]
    return gpd.GeoDataFrame(geometry=data, crs="EPSG:4326")

@pytest.fixture
def mock_clusters():
    """
    Create mock cluster data for TRACLUS visualization tests.
    """
    clusters = [
        [np.array([[0, 0], [1, 1], [2, 2]])],
        [np.array([[2, 2], [3, 3], [4, 4]])],
        [np.array([[4, 4], [5, 5], [6, 6]])]
    ]
    cluster_assignments = [1, 2, 3]
    return clusters, cluster_assignments

def test_map_ilustration(mock_geodataframe):
    """
    Test the map_ilustration function for generating a trajectory map.
    """
    gdf = mock_geodataframe
    minx, miny, maxx, maxy = 0, 0, 6, 6  # Define area of interest
    img_data = map_ilustration(gdf, minx, miny, maxx, maxy)
    assert isinstance(img_data, BytesIO), "Output should be a BytesIO object."
    assert img_data.getbuffer().nbytes > 0, "Map image data should not be empty."

def test_map_heat(mock_geodataframe):
    """
    Test the map_heat function for generating a heatmap of trajectory density.
    """
    gdf = mock_geodataframe
    minx, miny, maxx, maxy = 0, 0, 6, 6  # Define area of interest
    img_data = map_heat(gdf, minx, miny, maxx, maxy)
    assert isinstance(img_data, BytesIO), "Output should be a BytesIO object."
    assert img_data.getbuffer().nbytes > 0, "Heatmap image data should not be empty."

def test_plot_map_traclus():
    """
    Test the plot_map_traclus function for generating representative trajectory maps.
    """
    representative_clusters = [
        [(0, 0), (1, 1), (2, 2)], 
        [(3, 3), (4, 4), (5, 5)],
        [(6, 6), (7, 7), (8, 8)]
    ]
    img_data = plot_map_traclus(representative_clusters)
    assert isinstance(img_data, BytesIO), "Output should be a BytesIO object."
    assert img_data.getbuffer().nbytes > 0, "Trajectory map image data should not be empty."

def test_plot_clusters_on_map(mock_clusters):
    """
    Test the plot_clusters_on_map function for generating maps with clusters.
    """
    clusters, _ = mock_clusters
    img_data = plot_clusters_on_map(clusters, descart_minun_one=True)
    assert isinstance(img_data, BytesIO), "Output should be a BytesIO object."
    assert img_data.getbuffer().nbytes > 0, "Cluster map image data should not be empty."

def test_plot_segments_on_map(mock_clusters):
    """
    Test the plot_segments_on_map function for generating maps with individual segments.
    """
    clusters, cluster_assignments = mock_clusters
    segments = [segment for cluster in clusters for segment in cluster]
    img_data = plot_segments_on_map(segments, cluster_assignments, descart_minun_one=False)
    assert isinstance(img_data, BytesIO), "Output should be a BytesIO object."
    assert img_data.getbuffer().nbytes > 0, "Segment map image data should not be empty."
