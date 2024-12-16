import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import LineString
import base64
import io

def load_and_simplify_data(filename, rows, tolerance=0.001):
    """
    Loads and simplifies data from a file, either CSV or Excel, and prepares it for processing.

    Parameters:
    - filename: str, the uploaded file containing trajectory data.
    - rows: int, the maximum number of rows to load.
    - tolerance: float, the tolerance value used to simplify geometries.

    Returns:
    - gdf: GeoDataFrame, geospatial data with simplified geometries.
    - trayectorias: list, a list of numpy arrays representing trajectories.
    - df: DataFrame, original data with converted geometries and filtered rows.
    """
    try:
        # Decode the base64-encoded file content
        _, content_string = filename.split(',')
        decoded = base64.b64decode(content_string)

        # Load data as DataFrame based on file type
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=rows)
        else:
            df = pd.read_excel(io.BytesIO(decoded), nrows=rows)

        # Convert the POLYLINE column to geometries
        def create_line(x):
            points = json.loads(x)
            if len(points) > 1:
                return LineString(points)
            return None

        # Apply geometry transformation and filter invalid rows
        df['geometry'] = df['POLYLINE'].apply(create_line)
        df = df[df['geometry'].notnull()]
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Simplify geometries based on the tolerance value
        gdf['geometry'] = gdf['geometry'].simplify(tolerance)

        # Convert POLYLINE JSON strings to lists and filter invalid entries
        df['POLYLINE'] = df['POLYLINE'].apply(lambda x: json.loads(x) if pd.notnull(x) else None)
        df = df[df['POLYLINE'].apply(lambda x: x is not None and len(x) > 0)]

        # Prepare trajectories for TRACLUS processing
        trayectorias = [np.array(polyline) for polyline in df['POLYLINE']]

        return gdf, trayectorias, df

    except Exception as e:
        # Handle errors in data loading
        print(f"Error loading and simplifying data: {e}")
        raise e


def is_segment_in_trajectory(segment, trajectory):
    """
    Checks if a segment intersects a given trajectory.

    Parameters:
    - segment: list, a line segment represented as a list of points.
    - trajectory: list, a trajectory represented as a list of points.

    Returns:
    - bool: True if the segment intersects the trajectory, False otherwise.
    """
    segment_line = LineString(segment)
    trajectory_line = LineString(trajectory)
    return segment_line.intersects(trajectory_line)


def relational_table(df, segments, cluster_assignments, representative_trajectories):
    """
    Creates a relational table that associates segments with trajectories, clusters, and representatives.

    Parameters:
    - df: DataFrame, the input data.
    - segments: list, a list of segments as LineString objects.
    - cluster_assignments: list, cluster IDs assigned to each segment.
    - representative_trajectories: list, representative trajectories for each cluster.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing segment relationships with clusters and representatives.
    """
    gdf_stc_data = []
    i = 0  # Keeps track of the last processed trajectory index

    for segment, cluster_id in zip(segments, cluster_assignments):
        line = LineString(segment)
        tray_id = -1  # Default indicator for "not found"
        line_index = None

        # Check which trajectory contains the segment
        for index, trajectory in zip(df.index, df['POLYLINE']):
            if is_segment_in_trajectory(segment, trajectory) and i <= index:
                line_index = index
                i = index
                break

        # Determine if the segment intersects any representative trajectory
        for rep_id, rep_trajectory in enumerate(representative_trajectories):
            rep_line = LineString(rep_trajectory)
            if rep_line.intersects(line):
                tray_id = rep_id
                break

        # Append the relational data for the segment
        gdf_stc_data.append({
            'line_index': line_index,
            'segment': line,
            'cluster_id': cluster_id,
            'represent_tray_id': tray_id
        })

    # Create and return a GeoDataFrame
    gdf_stc = gpd.GeoDataFrame(gdf_stc_data, columns=['line_index', 'segment', 'cluster_id', 'represent_tray_id'])
    return gdf_stc


def get_cluster_graph(cluster_assignments):
    """
    Creates a filtered list of cluster assignments for visualization.

    Parameters:
    - cluster_assignments: list, cluster IDs assigned to each segment.

    Returns:
    - list: Cluster IDs excluding outliers (represented by -1).
    """
    return [asig for asig in cluster_assignments if asig != -1]
