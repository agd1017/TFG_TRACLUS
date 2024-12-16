import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import contextily as ctx
from io import BytesIO
from pyproj import Transformer
from shapely.geometry import LineString, box

#* Utility Functions for Maps and TRACLUS Visualization

def get_coordinates(gdf):
    """
    Get the minimum and maximum x and y coordinates from a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame containing geometry.

    Returns:
        tuple: (minx, miny, maxx, maxy) - minimum and maximum coordinates.
    """
    x_coords_flat, y_coords_flat = list_coordinates(gdf)
    maxx, maxy, minx, miny = max(x_coords_flat), max(y_coords_flat), min(x_coords_flat), min(y_coords_flat)
    return minx, miny, maxx, maxy

def filter_data_in_area(gdf, minx, miny, maxx, maxy):
    """
    Filter GeoDataFrame to include geometries within a specified area.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.
        minx, miny, maxx, maxy (float): Coordinates defining the area.

    Returns:
        GeoDataFrame: Filtered GeoDataFrame.
    """
    area_of_interest = box(minx, miny, maxx, maxy)
    gdf_filtered = gdf[gdf.geometry.within(area_of_interest)]
    return gdf_filtered

def list_coordinates(gdf):
    """
    Extract x and y coordinates from a GeoDataFrame's LineString geometries.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame containing LineString geometries.

    Returns:
        tuple: Two lists containing x and y coordinates respectively.
    """
    x_coords_flat = []
    y_coords_flat = []

    for geom in gdf['geometry']:
        if isinstance(geom, LineString):
            for point in geom.coords:
                x_coords_flat.append(point[0])
                y_coords_flat.append(point[1])

    return x_coords_flat, y_coords_flat

def map_ilustration(gdf, minx, miny, maxx, maxy):
    """
    Create a map with trajectories overlaid on a base map.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with trajectories.
        minx, miny, maxx, maxy (float): Coordinates for filtering data.

    Returns:
        BytesIO: Image data of the generated map.
    """
    gdf = gdf.set_crs("EPSG:4326")
    gdf = filter_data_in_area(gdf, minx, miny, maxx, maxy)
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    gdf.plot(ax=ax, linewidth=0.5, color='green')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    plt.title('Map of Trajectories with Background Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close(fig)

    return img_data

def map_heat(gdf, minx, miny, maxx, maxy, bin_count=300):
    """
    Create a heatmap of trajectory density.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with trajectories.
        minx, miny, maxx, maxy (float): Coordinates for filtering data.
        bin_count (int): Number of bins for the heatmap.

    Returns:
        BytesIO: Image data of the generated heatmap.
    """
    x_coords_flat, y_coords_flat = list_coordinates(gdf)
    heatmap, _, _ = np.histogram2d(x_coords_flat, y_coords_flat, bins=bin_count, density=True, range=[[minx, maxx], [miny, maxy]])

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    xmin, ymin = transformer.transform(minx, miny)
    xmax, ymax = transformer.transform(maxx, maxy)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    norm = colors.LogNorm(vmin=heatmap.min() + 1, vmax=heatmap.max())
    mappable = ax.imshow(heatmap.T, origin='lower', norm=norm, extent=[xmin, xmax, ymin, ymax], aspect='auto', alpha=0.7, zorder=2)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')

    plt.colorbar(mappable, label='Density')
    plt.title('Trajectory Heatmap with Background Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close(fig)

    return img_data

#* TRACLUS comparation mapas

def create_gdf(data):
    """
    Create a GeoDataFrame from a list of trajectory data.

    Args:
        data (list): List of trajectory data.

    Returns:
        GeoDataFrame: GeoDataFrame containing LineString geometries.
    """
    valid_geometries = [LineString(x) for x in data if len(x) > 1]
    gdf = gpd.GeoDataFrame(geometry=valid_geometries, crs='EPSG:4326')
    return gdf

def plot_map_traclus(representative_clusters, cmap='tab20'):
    """
    Plot TRACLUS representative trajectories on a map.

    Args:
        representative_clusters (list): List of representative trajectories.
        cmap (str): Colormap for clusters.

    Returns:
        BytesIO: Image data of the plotted map.
    """
    gdf = create_gdf(representative_clusters)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    gdf.plot(ax=ax, cmap=cmap, linewidth=2)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string())

    plt.title('Representation of TRACLUS Trajectories')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close(fig)

    return img_data

def plot_clusters_on_map(clusters, descart_minun_one=True, cmap='tab20'):
    """
    Plots the clusters on a map using the provided cluster data.

    Args:
        clusters (list): A list of clusters, where each cluster is a list of segments.
        descart_minun_one (bool): If True, skips the cluster with ID -1.
        cmap (str): Colormap name for assigning colors to clusters.

    Returns:
        BytesIO: A BytesIO object containing the image data of the map.
    """
    # Create an empty GeoDataFrame to store geometry and cluster IDs
    gdf = gpd.GeoDataFrame(columns=['geometry', 'cluster_id'])

    cluster_id = 0
    # Iterate over each cluster
    for segment in clusters:
        if descart_minun_one and cluster_id == 0:
            # Skip the cluster with ID -1 if specified
            cluster_id += 1
            continue
        else:
            cluster_id += 1
        for part in segment:
            # Add each segment as a LineString geometry
            if isinstance(part, np.ndarray):  # Check if the segment is a NumPy array
                line = LineString(part)
                gdf = pd.concat([gdf, gpd.GeoDataFrame({'geometry': [line], 'cluster_id': [cluster_id]})], ignore_index=True)

    # Set the coordinate reference system to EPSG:4326
    gdf.crs = "EPSG:4326"

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    if not gdf.empty:
        # Generate colors for each cluster using the provided colormap
        unique_clusters = gdf['cluster_id'].unique()
        colormap = plt.cm.get_cmap(cmap, len(unique_clusters))  # Use the desired colormap
        color_dict = {cid: colormap(i) for i, cid in enumerate(unique_clusters)}

        # Plot each segment using its cluster color
        for _, row in gdf.iterrows():
            color = color_dict[row['cluster_id']]
            gdf.loc[gdf.index == _, 'geometry'].plot(ax=ax, linewidth=2, color=color)

        # Add a basemap with the same CRS
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string())
    else:
        print("No valid data to plot.")

    # Add title and labels
    plt.title('Representation of TRACLUS Clusters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Ensure the axis is visible
    ax.set_axis_on()

    # Save the plot to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    # Close the figure to free memory
    plt.close(fig)

    return img_data


def plot_segments_on_map(segments, cluster_assignments, descart_minun_one=False, cmap='tab20'):
    """
    Plots individual segments on a map and colors them based on their cluster assignments.

    Args:
        segments (list): List of segments (each as NumPy arrays).
        cluster_assignments (list): List of cluster IDs corresponding to each segment.
        descart_minun_one (bool): If True, skips segments with cluster ID -1.
        cmap (str): Colormap name for assigning colors to clusters.

    Returns:
        BytesIO: A BytesIO object containing the image data of the map.
    """
    # Create an empty GeoDataFrame to store geometry and cluster IDs
    gdf = gpd.GeoDataFrame(columns=['geometry', 'cluster_id'])

    # Add each segment to the GeoDataFrame
    for segment, cluster_id in zip(segments, cluster_assignments):
        if descart_minun_one and cluster_id == -1:
            # Skip segments with cluster ID -1 if specified
            continue
        if isinstance(segment, np.ndarray):  # Check if the segment is a NumPy array
            line = LineString(segment)
            gdf = pd.concat([gdf, gpd.GeoDataFrame({'geometry': [line], 'cluster_id': [cluster_id]})], ignore_index=True)

    # Set the coordinate reference system to EPSG:4326
    gdf.crs = "EPSG:4326"

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    if not gdf.empty:
        # Generate colors for each cluster
        unique_clusters = gdf['cluster_id'].unique()
        colormap = plt.cm.get_cmap(cmap, len(unique_clusters))  # Use the desired colormap
        color_dict = {cid: colormap(i) for i, cid in enumerate(unique_clusters)}

        # Plot each segment using its cluster color
        for _, row in gdf.iterrows():
            color = color_dict[row['cluster_id']]
            gdf.loc[gdf.index == _, 'geometry'].plot(ax=ax, linewidth=2, color=color)

        # Add a basemap with the same CRS
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string())
    else:
        print("No valid data to plot.")

    # Add title and labels
    plt.title('Representation of TRACLUS Segments')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Ensure the axis is visible
    ax.set_axis_on()

    # Save the plot to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    # Close the figure to free memory
    plt.close(fig)

    return img_data