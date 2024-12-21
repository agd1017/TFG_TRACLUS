import os
import shutil
import pandas as pd
import geopandas as gpd

from app.utils.config import UPLOAD_FOLDER

# Save data to a folder

def save_html_or_binary(file_path, content):
    """
    Saves the given content to a file, either as text or binary.
    If content is in bytes or BytesIO, it's saved as binary; otherwise, it's saved as text.
    """
    if isinstance(content, bytes):
        with open(file_path, 'wb') as f:
            f.write(content)
    elif hasattr(content, 'getvalue'):  # If it's a BytesIO object
        with open(file_path, 'wb') as f:
            f.write(content.getvalue())
    else:
        # If it's a string, save as text
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

def save_data(folder_name, gdf=None, html_map=None, html_heatmap=None, 
            traclus_map_optics=None, traclus_map_segments_optics=None, traclus_map_cluster_optics=None, tabla_optics=None, graph_optics=None,
            traclus_map_hdbscan=None, traclus_map_segments_hdbscan=None, traclus_map_cluster_hdbscan=None, tabla_hdbscan=None, graph_hdbscan=None,
            traclus_map_dbscan=None, traclus_map_segments_dbscan=None, traclus_map_cluster_dbscan=None, tabla_dbscan=None, graph_dbscan=None,
            traclus_map_spect=None, traclus_map_segments_spect=None, traclus_map_cluster_spect=None, tabla_spect=None, graph_spect=None,
            traclus_map_aggl=None, traclus_map_segments_aggl=None, traclus_map_cluster_aggl=None, tabla_aggl=None, graph_aggl=None):
    """
    Saves multiple types of data (GeoDataFrame, HTML maps, tables, graphs) to a specific folder.
    It creates a new folder and saves the provided data inside it.
    If any data is None, it is skipped.
    """
    folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
    
    # Delete the folder if it already exists, and create it again
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Deletes the folder and its contents
    
    # Create the folder again
    os.makedirs(folder_path)

    # Save data to the folder:
    # 1. Save GeoDataFrame as GeoJSON
    if gdf is not None:
        gdf.to_file(os.path.join(folder_path, "resultado_gdf.geojson"), driver='GeoJSON')

    # 2. Save HTML maps if provided
    save_html_or_binary(os.path.join(folder_path, "html_map.html"), html_map if html_map is not None else "")
    save_html_or_binary(os.path.join(folder_path, "html_heatmap.html"), html_heatmap if html_heatmap is not None else "")

    # 3. Save TRACLUS maps in HTML format if provided
    if traclus_map_optics is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_optics.html"), traclus_map_optics)

    if traclus_map_hdbscan is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_hdbscan.html"), traclus_map_hdbscan)

    if traclus_map_dbscan is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_dbscan.html"), traclus_map_dbscan)

    if traclus_map_spect is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_spectralclustering.html"), traclus_map_spect)

    if traclus_map_aggl is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_agglomerativeclustering.html"), traclus_map_aggl)

    # 4. Save segment maps as HTML if provided
    if traclus_map_segments_optics is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_segments_optics.html"), traclus_map_segments_optics)

    if traclus_map_segments_hdbscan is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_segments_hdbscan.html"), traclus_map_segments_hdbscan)

    if traclus_map_segments_dbscan is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_segments_dbscan.html"), traclus_map_segments_dbscan)

    if traclus_map_segments_spect is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_segments_spectralclustering.html"), traclus_map_segments_spect)

    if traclus_map_segments_aggl is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_segments_agglomerativeclustering.html"), traclus_map_segments_aggl)

    # 5. Save cluster maps as HTML if provided
    if traclus_map_cluster_optics is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_cluster_optics.html"), traclus_map_cluster_optics)

    if traclus_map_cluster_hdbscan is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_cluster_hdbscan.html"), traclus_map_cluster_hdbscan)

    if traclus_map_cluster_dbscan is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_cluster_dbscan.html"), traclus_map_cluster_dbscan)

    if traclus_map_cluster_spect is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_cluster_spectralclustering.html"), traclus_map_cluster_spect)

    if traclus_map_cluster_aggl is not None:
        save_html_or_binary(os.path.join(folder_path, "traclus_map_cluster_agglomerativeclustering.html"), traclus_map_cluster_aggl)

    # 6. Save algorithm-generated tables as CSV if provided
    if tabla_optics is not None:
        tabla_optics.to_csv(os.path.join(folder_path, "tabla_optics.csv"), index=False)

    if tabla_hdbscan is not None:
        tabla_hdbscan.to_csv(os.path.join(folder_path, "tabla_hdbscan.csv"), index=False)

    if tabla_dbscan is not None:
        tabla_dbscan.to_csv(os.path.join(folder_path, "tabla_dbscan.csv"), index=False)

    if tabla_spect is not None:
        tabla_spect.to_csv(os.path.join(folder_path, "tabla_spectralclustering.csv"), index=False)

    if tabla_aggl is not None:
        tabla_aggl.to_csv(os.path.join(folder_path, "tabla_agglomerativeclustering.csv"), index=False)

    # 7. Save graphs as CSV if provided
    if graph_optics is not None:
        pd.DataFrame({'Data': graph_optics}).to_csv(os.path.join(folder_path, "graph_optics.csv"), index=False)

    if graph_hdbscan is not None:
        pd.DataFrame({'Data': graph_hdbscan}).to_csv(os.path.join(folder_path, "graph_hdbscan.csv"), index=False)

    if graph_dbscan is not None:
        pd.DataFrame({'Data': graph_dbscan}).to_csv(os.path.join(folder_path, "graph_dbscan.csv"), index=False)

    if graph_spect is not None:
        pd.DataFrame({'Data': graph_spect}).to_csv(os.path.join(folder_path, "graph_spectralclustering.csv"), index=False)

    if graph_aggl is not None:
        pd.DataFrame({'Data': graph_aggl}).to_csv(os.path.join(folder_path, "graph_agglomerativeclustering.csv"), index=False)

# Load data from a folder

def read_html_file(file_path):
    """
    Reads an HTML file from the given path.
    Returns the content either as a string or as bytes if there's a decoding error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'rb') as f:
            return f.read()  # Returns as bytes

def convert_to_dataframe(file_path):
    """
    Converts a CSV file to a Pandas DataFrame.
    """
    return pd.read_csv(file_path)

def load_data(files, folder_name):
    """
    Loads data from files in the specified folder.
    Returns various data objects like GeoDataFrame, HTML maps, tables, and graphs based on available files.
    """
    # Reset global variables
    optics_on = dbscan_on = hdbscan_on = aggl_on = spect_on  = False
    gdf = html_map = html_heatmap = None
    traclus_map_optics = traclus_map_cluster_optics = traclus_map_segments_optics = tabla_optics = graph_optics  = None
    traclus_map_hdbscan = traclus_map_cluster_hdbscan = traclus_map_segments_hdbscan = tabla_hdbscan = graph_hdbscan = None
    traclus_map_dbscan = traclus_map_cluster_dbscan = traclus_map_segments_dbscan = tabla_dbscan = graph_dbscan = None
    traclus_map_spect = traclus_map_cluster_spect = traclus_map_segments_spect = tabla_spect = graph_spect = None
    traclus_map_aggl = traclus_map_cluster_aggl = traclus_map_segments_aggl = tabla_aggl = graph_aggl = None
    
    for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, folder_name, file)

            if file == "resultado_gdf.geojson":
                gdf = gpd.read_file(file_path)
            elif file == "html_map.html":
                html_map = read_html_file(file_path)
            elif file == "html_heatmap.html":
                html_heatmap = read_html_file(file_path)
            elif file == "traclus_map_optics.html":
                optics_on = True
                traclus_map_optics = read_html_file(file_path)
            elif file == "traclus_map_cluster_optics.html":
                traclus_map_cluster_optics = read_html_file(file_path)
            elif file == "traclus_map_segments_optics.html":
                traclus_map_segments_optics = read_html_file(file_path)
            elif file == "traclus_map_hdbscan.html":
                hdbscan_on = True
                traclus_map_hdbscan = read_html_file(file_path)
            elif file == "traclus_map_cluster_hdbscan.html":
                traclus_map_cluster_hdbscan = read_html_file(file_path)
            elif file == "traclus_map_segments_hdbscan.html":
                traclus_map_segments_hdbscan = read_html_file(file_path)
            elif file == "traclus_map_dbscan.html":
                dbscan_on = True
                traclus_map_dbscan = read_html_file(file_path)
            elif file == "traclus_map_segments_dbscan.html":
                traclus_map_cluster_dbscan = read_html_file(file_path)
            elif file == "traclus_map_cluster_dbscan.html":
                traclus_map_cluster_dbscan = read_html_file(file_path)
            elif file == "traclus_map_spectralclustering.html":
                spect_on = True
                traclus_map_spect = read_html_file(file_path)
            elif file == "traclus_map_segments_spectralclustering.html":
                traclus_map_segments_spect = read_html_file(file_path)
            elif file == "traclus_map_cluster_spectralclustering.html":
                traclus_map_cluster_spect = read_html_file(file_path)
            elif file == "traclus_map_agglomerativeclustering.html":
                aggl_on = True
                traclus_map_aggl = read_html_file(file_path)
            elif file == "traclus_map_segments_agglomerativeclustering.html":
                traclus_map_segments_aggl = read_html_file(file_path)
            elif file == "traclus_map_cluster_agglomerativeclustering.html":
                traclus_map_cluster_aggl= read_html_file(file_path)
            elif file == "tabla_optics.csv":
                tabla_optics = convert_to_dataframe(file_path)
            elif file == "tabla_hdbscan.csv":
                tabla_hdbscan = convert_to_dataframe(file_path)
            elif file == "tabla_dbscan.csv":
                tabla_dbscan = convert_to_dataframe(file_path)
            elif file == "tabla_spectralclustering.csv":
                tabla_spect = convert_to_dataframe(file_path)
            elif file == "tabla_agglomerativeclustering.csv":
                tabla_aggl= convert_to_dataframe(file_path)
            elif file == "graph_optics.csv":
                graph_optics = pd.read_csv(file_path)['Data'].tolist()
            elif file == "graph_hdbscan.csv":
                graph_hdbscan = pd.read_csv(file_path)['Data'].tolist()
            elif file == "graph_dbscan.csv":
                graph_dbscan = pd.read_csv(file_path)['Data'].tolist()
            elif file == "graph_spectralclustering.csv":
                graph_spect = pd.read_csv(file_path)['Data'].tolist()
            elif file == "graph_agglomerativeclustering.csv":
                graph_aggl = pd.read_csv(file_path)['Data'].tolist()

    return optics_on, dbscan_on, hdbscan_on, aggl_on, spect_on, \
            gdf, html_map, html_heatmap, \
            traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics, \
            traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan, \
            traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan, \
            traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect, \
            traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl