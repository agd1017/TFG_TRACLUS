import dash
from dash import html, dcc, callback_context
import zipfile
import shutil
import os
import io
import plotly.express as px
from collections import Counter
import time

from utils.config import UPLOAD_FOLDER
from utils.data_saveload import save_data, load_data
from utils.data_utils import list_files_in_folder
from views.layout.navbar import get_navbar
from views.layout.dataupload_page import get_page_dataupdate
from views.layout.experiment_page import get_page_experiment
from views.layout.map_page import get_page_map, get_map_image_as_html
from views.layout.select_page import get_page_select
from views.layout.TRACLUSmap_page import get_page_maptraclus, get_clusters_map
from views.layout.table_page import get_page_tables, get_table
from controllers.clustering import data_constructor

# -- Callbacks section --

# Function to display the page based on the pathname
def display_page(pathname):
    """
    Returns the appropriate page layout based on the provided pathname.
    """
    if pathname == '/':
        return get_page_select()  # Returns the page for selecting options
    elif pathname == '/new-experiment':
        return get_page_experiment()  # Returns the page for a new experiment
    elif pathname == '/data-update':
        return get_page_dataupdate()  # Returns the page for data updates
    elif pathname == '/map-page':
        return get_page_map()  # Returns the map page layout
    elif pathname == '/TRACLUS-map':
        return get_page_maptraclus(optics_on, hdbscan_on, dbscan_on, spect_on, aggl_on)  # Returns the TRACLUS map page
    elif pathname == '/estadisticas':
        return get_page_tables(optics_on, hdbscan_on, dbscan_on, spect_on, aggl_on)  # Returns statistics page
    else:
        return get_page_select()  # Default to the page for selecting options

# -- Callbacks for navbar --
def update_navbar(pathname):
    """
    Updates the navigation bar based on the current pathname.
    """
    return get_navbar(pathname)

# Callback to download data as a ZIP file
def download_data(n_clicks):
    """
    Creates a ZIP file with relevant data (images and text files) and sends it for download.
    """
    # Create a ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, mode="w") as zf:
        # Dictionaries for image files and text files
        images = {
            "map.png": html_map,
            "heatmap.png": html_heatmap
        }
        txt_files = {}

        # Add text files and images based on activated options
        if optics_on:
            txt_files["table_optics.csv"] = table_optics.to_csv(index=False, sep='\t')
            images["traclus_map_optics.png"] = traclus_map_optics
            images["traclus_map_segments_optics.png"] = traclus_map_segments_optics
            images["traclus_map_cluster_optics.png"] = traclus_map_cluster_optics
        if hdbscan_on:
            txt_files["table_hdbscan.csv"] = table_hdbscan.to_csv(index=False, sep='\t')
            images["traclus_map_hdbscan.png"] = traclus_map_hdbscan
            images["traclus_map_segments_hdbscan.png"] = traclus_map_segments_hdbscan
            images["traclus_map_cluster_hdbscan.png"] = traclus_map_cluster_hdbscan
        if dbscan_on:
            txt_files["table_dbscan.csv"] = table_dbscan.to_csv(index=False, sep='\t')
            images["traclus_map_dbscan.png"] = traclus_map_dbscan
            images["traclus_map_segments_dbscan.png"] = traclus_map_segments_dbscan
            images["traclus_map_cluster_dbscan.png"] = traclus_map_cluster_dbscan
        if spect_on:
            txt_files["table_spectralclustering.csv"] = table_spect.to_csv(index=False, sep='\t')
            images["traclus_map_spectralclustering.png"] = traclus_map_spect
            images["traclus_map_segments_spectralclustering.png"] = traclus_map_segments_spect
            images["traclus_map_cluster_spectralclustering.png"] = traclus_map_cluster_spect
        if aggl_on:
            txt_files["table_agglomerativeclustering.csv"] = table_aggl.to_csv(index=False, sep='\t')
            images["traclus_map_agglomerativeclustering.png"] = traclus_map_aggl
            images["traclus_map_segments_agglomerativeclustering.png"] = traclus_map_segments_aggl
            images["traclus_map_cluster_agglomerativeclustering.png"] = traclus_map_cluster_aggl

        # Add text files to ZIP
        for filename, txt_content in txt_files.items():
            zf.writestr(filename, txt_content)

        # Ensure images are BytesIO objects and add them to the ZIP
        for img_name, img_data in images.items():
            if not isinstance(img_data, io.BytesIO):  # Convert image to BytesIO if it's not already
                img_buffer = io.BytesIO(img_data)
            else:
                img_buffer = img_data
            img_buffer.seek(0)  # Ensure pointer is at the start of the buffer
            zf.writestr(img_name, img_buffer.read())  # Add image to ZIP

    # Reset pointer for reading content
    zip_buffer.seek(0)

    # Return the ZIP file for download
    return dcc.send_bytes(zip_buffer.getvalue(), "table.zip")

# -- Callbacks for select page --
def navigate_experiment_page(n_clicks_new):
    """
    Navigates to the new experiment page if a new experiment button is clicked.
    """
    if n_clicks_new > 0:
        return '/new-experiment'
    return '/'

def display_files_in_selected_folder(folder_name, n_clicks_previous, is_modal_open):
    """
    Displays the files from the selected folder if any button is clicked.
    """
    if n_clicks_previous > 0:
        if folder_name is None:
            return dash.no_update, not is_modal_open

        # Get files in the selected folder
        files = list_files_in_folder(folder_name)

        load = load_data(files, folder_name)

        # Global variables for data and images
        global optics_on, dbscan_on, hdbscan_on, aggl_on, spect_on
        global gdf, html_map, html_heatmap
        global traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, table_optics, graph_optics
        global traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, table_hdbscan, graph_hdbscan
        global traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, table_dbscan, graph_dbscan
        global traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, table_spect, graph_spect
        global traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, table_aggl, graph_aggl
        
        optics_on, dbscan_on, hdbscan_on, aggl_on, spect_on, \
        gdf, html_map, html_heatmap, \
        traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, table_optics, graph_optics, \
        traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, table_hdbscan, graph_hdbscan, \
        traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, table_dbscan, graph_dbscan, \
        traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, table_spect, graph_spect, \
        traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, table_aggl, graph_aggl = load

        # Check if any processing option is on
        if optics_on or dbscan_on or hdbscan_on or aggl_on or spect_on:
            return '/map-page', is_modal_open
    
    return dash.no_update, is_modal_open

def toggle_modal(n_delete, n_cancel, n_confirm, is_open, folder_name):
    """
    Toggles the modal window when certain actions (delete, cancel, confirm) are triggered.
    """
    if folder_name is not None and (n_delete or n_cancel or n_confirm):
        return not is_open
    return is_open

def delete_experiment(n_clicks, folder_name):
    """
    Deletes the selected experiment folder if the delete button is clicked.
    """
    if n_clicks and folder_name:
        folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Remove the folder and its contents
            return "/"
    return dash.no_update

# -- Callbacks for experiment page --

def navigate_to_page_dataupdate(n_clicks_data, checkoptics, optics_metric_value, optics_algorithm_value, optics_eps_value, optics_sample_value, checkdbscan, 
                                dbscan_metric_value, dbscan_algorithm_value, dbscan_eps_value, dbscan_sample_value, checkhdbscan, hdbscan_metric_value, 
                                hdbscan_algorithm_value, hdbscan_sample_value, checkagglomerativeclustering, aggl_metric_value, aggl_linkage_value, 
                                aggl_n_clusters_value, checkspectralclustering, spect_affinity_value, spect_assign_labels_value, spect_n_clusters_value,
                                is_open):
    """
    Updates the experiment parameters and navigates to the data update page if the conditions are met.
    """
    if n_clicks_data is not None and n_clicks_data > 0:
        global optics_on, dbscan_on, hdbscan_on, aggl_on, spect_on 
        optics_on = dbscan_on = hdbscan_on = aggl_on = spect_on = False
        
        global optics_metric, optics_algorithm, optics_eps, optics_sample
        optics_metric = optics_algorithm = optics_eps = optics_sample = None
        global dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample
        dbscan_metric = dbscan_algorithm = dbscan_eps = dbscan_sample = None
        global hdbscan_metric, hdbscan_algorithm, hdbscan_sample
        hdbscan_metric = hdbscan_algorithm = hdbscan_sample = None
        global aggl_metric, aggl_linkage, aggl_n_clusters
        aggl_metric = aggl_linkage = aggl_n_clusters = None
        global spect_affinity, spect_assign_labels, spect_n_clusters
        spect_affinity = spect_assign_labels = spect_n_clusters = None

        global params
        params = {}

        if checkoptics and optics_metric_value and optics_algorithm_value and optics_eps_value and optics_sample_value:
            optics_on = True
            optics_metric = optics_metric_value
            optics_algorithm = optics_algorithm_value
            optics_eps = optics_eps_value
            optics_sample = optics_sample_value
            params['optics'] = [optics_metric, optics_algorithm, optics_eps, optics_sample]
        if checkdbscan and dbscan_metric_value and dbscan_algorithm_value and dbscan_eps_value and dbscan_sample_value:
            dbscan_on = True
            dbscan_metric = dbscan_metric_value
            dbscan_algorithm = dbscan_algorithm_value
            dbscan_eps = dbscan_eps_value
            dbscan_sample = dbscan_sample_value
            params['dbscan'] = [dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample]
        if checkhdbscan and hdbscan_metric_value and hdbscan_algorithm_value and hdbscan_sample_value:
            hdbscan_on = True
            hdbscan_metric = hdbscan_metric_value
            hdbscan_algorithm = hdbscan_algorithm_value
            hdbscan_sample = hdbscan_sample_value
            params['hdbscan'] = [hdbscan_metric, hdbscan_algorithm, hdbscan_sample]
        if checkagglomerativeclustering and aggl_metric_value and aggl_linkage_value and aggl_n_clusters_value:
            aggl_on = True
            aggl_metric = aggl_metric_value
            aggl_linkage = aggl_linkage_value
            aggl_n_clusters = aggl_n_clusters_value
            params['agglomerativeclustering'] = [aggl_metric, aggl_linkage, aggl_n_clusters]
        if checkspectralclustering and spect_affinity_value and spect_assign_labels_value and spect_n_clusters_value:
            spect_on = True
            spect_affinity = spect_affinity_value
            spect_assign_labels = spect_assign_labels_value
            spect_n_clusters = spect_n_clusters_value
            params['spectralclustering'] = [spect_affinity, spect_assign_labels, spect_n_clusters]

        if optics_on or dbscan_on or hdbscan_on or aggl_on or spect_on:
            return '/data-update', is_open
        return dash.no_update, not is_open

    return '/new-experiment', is_open

# Function to toggle controls for the 'rowo' component based on the selector value
def toggle_rowo_controls(selector_value_o):
    """
    Toggles the enabled/disabled state of the 'rowo' component controls based on the selector value.
    If the value contains 'on', the controls are enabled, otherwise, they are disabled.
    """
    is_enabled = 'on' in selector_value_o
    return not is_enabled, not is_enabled, not is_enabled, not is_enabled

# Function to toggle controls for the 'rowd' component based on the selector value
def toggle_rowd_controls(selector_value_d):
    """
    Toggles the enabled/disabled state of the 'rowd' component controls based on the selector value.
    If the value contains 'on', the controls are enabled, otherwise, they are disabled.
    """
    is_enabled = 'on' in selector_value_d
    return not is_enabled, not is_enabled, not is_enabled, not is_enabled

# Function to toggle controls for the 'rowh' component based on the selector value
def toggle_rowh_controls(selector_value_h):
    """
    Toggles the enabled/disabled state of the 'rowh' component controls based on the selector value.
    If the value contains 'on', the controls are enabled, otherwise, they are disabled.
    """
    is_enabled = 'on' in selector_value_h
    return not is_enabled, not is_enabled, not is_enabled

# Function to toggle controls for the 'rowa' component based on the selector value
def toggle_rowa_controls(selector_value_a):
    """
    Toggles the enabled/disabled state of the 'rowa' component controls based on the selector value.
    If the value contains 'on', the controls are enabled, otherwise, they are disabled.
    """
    is_enabled = 'on' in selector_value_a
    return not is_enabled, not is_enabled, not is_enabled

# Function to toggle controls for the 'rows' component based on the selector value
def toggle_rows_controls(selector_value_s):
    """
    Toggles the enabled/disabled state of the 'rows' component controls based on the selector value.
    If the value contains 'on', the controls are enabled, otherwise, they are disabled.
    """
    is_enabled = 'on' in selector_value_s
    return not is_enabled, not is_enabled, not is_enabled

# -- Callbacks for data upload page --

def process_csv_from_url(n_clicks_upload, data, nrows, folder_name):
    """
    Processes the uploaded CSV data and triggers the TRACLUS clustering algorithm.
    The function validates the inputs and starts the clustering process based on the chosen algorithm.
    """
    if n_clicks_upload is not None and n_clicks_upload > 0:
        if not folder_name:
            return dash.no_update, html.Div(["Por favor, introduce un nombre para el experimento."])
        if not data:
            return dash.no_update, html.Div(["No se ha introducido ningún enlace."])
        if not nrows:
            return dash.no_update, html.Div(["No se ha introducido el número de filas."])
        
        start_TRACLUS = time.time()
        result = data_constructor(data, nrows, optics_on, optics_metric, optics_algorithm, optics_eps, optics_sample, 
                                dbscan_on, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample, 
                                hdbscan_on, hdbscan_metric, hdbscan_algorithm, hdbscan_sample, 
                                aggl_on, aggl_metric, aggl_linkage, aggl_n_clusters, 
                                spect_on, spect_affinity, spect_assign_labels, spect_n_clusters)
        end_TRACLUS = time.time()

        print(f"Ejecution time: {end_TRACLUS - start_TRACLUS} segundos")

        # Storing the results from the clustering
        global gdf, tray, html_map, html_heatmap
        global traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, table_optics, graph_optics
        global traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, table_hdbscan, graph_hdbscan
        global traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, table_dbscan, graph_dbscan
        global traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, table_spect, graph_spect
        global traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, table_aggl, graph_aggl
        global error_message

        gdf, tray, html_map, html_heatmap, \
        traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, table_optics, graph_optics, \
        traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, table_hdbscan, graph_hdbscan, \
        traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, table_dbscan, graph_dbscan, \
        traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, table_spect, graph_spect, \
        traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, table_aggl, graph_aggl, \
        error_message = result

        # Handle error message if any
        if error_message:
            return dash.no_update, html.Div([error_message])
        
        # Save the results into a folder
        save_data(folder_name, gdf, html_map, html_heatmap, 
            traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, table_optics, graph_optics, 
            traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, table_hdbscan, graph_hdbscan, 
            traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, table_dbscan, graph_dbscan, 
            traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, table_spect, graph_spect, 
            traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, table_aggl, graph_aggl,
            params)

        return '/map-page', html.Div(['Procesamiento exitoso.'])

    # Return nothing if the button has not been clicked
    return dash.no_update, dash.no_update

# -- Callbacks for map page --

def update_map(*args):            
    """
    Updates the map on the map page.
    It fetches the current map image and heatmap HTML to display on the page.
    """
    map_image = get_map_image_as_html(html_map, html_heatmap)
    return [map_image]

# -- Callbacks for TRACLUS map page --

def display_clusters_1(*args):
    """
    Displays the clusters based on the triggered button.
    The clusters are fetched from the appropriate algorithm (Optics, HDBSCAN, DBSCAN, Spectral, Agglomerative).
    """
    ctx = callback_context

    if not ctx.triggered:
        if optics_on:
            return get_clusters_map(traclus_map_optics, traclus_map_cluster_optics,traclus_map_segments_optics)
        elif hdbscan_on:
            return get_clusters_map(traclus_map_hdbscan, traclus_map_cluster_hdbscan,traclus_map_segments_hdbscan)
        elif dbscan_on:
            return get_clusters_map(traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan)
        elif spect_on:
            return get_clusters_map(traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect)
        elif aggl_on:
            return get_clusters_map(traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl)
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'item-1-1':
            return get_clusters_map(traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics)
        elif button_id == 'item-1-2':
            return get_clusters_map(traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan)
        elif button_id == 'item-1-3':
            return get_clusters_map(traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan)
        elif button_id == 'item-1-4':
            return get_clusters_map(traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect)
        elif button_id == 'item-1-5':
            return get_clusters_map(traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl)

def display_clusters_2(*args):
    """
    Displays the clusters for the second group of buttons (similar to display_clusters_1).
    """
    ctx = callback_context

    if not ctx.triggered:
        if optics_on:
            return get_clusters_map(traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics)
        elif hdbscan_on:
            return get_clusters_map(traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan)
        elif dbscan_on:
            return get_clusters_map(traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan)
        elif spect_on:
            return get_clusters_map(traclus_map_spect, traclus_map_cluster_spect,traclus_map_segments_spect)
        elif aggl_on:
            return get_clusters_map(traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl)
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'item-2-1':
            return get_clusters_map(traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics)
        elif button_id == 'item-2-2':
            return get_clusters_map(traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan)
        elif button_id == 'item-2-3':
            return get_clusters_map(traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan)
        elif button_id == 'item-2-4':
            return get_clusters_map(traclus_map_spect, traclus_map_cluster_spect,traclus_map_segments_spect)
        elif button_id == 'item-2-5':
            return get_clusters_map(traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl)

# -- Callbacks for tables page --

def update_table(*args):
    """
    Updates the table data based on the selected filter.
    Displays the corresponding table for the selected clustering algorithm.
    """
    ctx = callback_context

    if not ctx.triggered:
        if optics_on:
            return get_table(table_optics)
        elif hdbscan_on:
            return get_table(table_hdbscan)
        elif dbscan_on:
            return get_table(table_dbscan)
        elif spect_on:
            return get_table(table_spect)
        elif aggl_on:
            return get_table(table_aggl)
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'table-1':
            return get_table(table_optics)
        elif button_id == 'table-2':
            return get_table(table_hdbscan)
        elif button_id == 'table-3':
            return get_table(table_dbscan)
        elif button_id == 'table-4':
            return get_table(table_spect)
        elif button_id == 'table-5':
            return get_table(table_aggl)

def update_graph(selected_filter):
    """
    Updates the graph based on the selected filter.
    If a valid filter is selected, it generates a bar chart for the selected clustering algorithm.
    """
    # If no filter is selected, return an empty graph
    if not selected_filter:
        return px.bar(
            title='Gráfico vacío',
            labels={'x': 'Clusters', 'y': 'Segments'}
        )

    # Filter the data based on the selected filter
    if selected_filter == 'optics' and optics_on:
        filtered_data = graph_optics
    elif selected_filter == 'hdbscan' and hdbscan_on:
        filtered_data = graph_hdbscan
    elif selected_filter == 'dbscan' and dbscan_on:
        filtered_data = graph_dbscan
    elif selected_filter == 'spectral' and spect_on:
        filtered_data = graph_spect
    elif selected_filter == 'agglomerative' and aggl_on:
        filtered_data = graph_aggl
    else:
        # If no data or the filter is disabled
        return px.bar(
            title=f'No hay datos para {selected_filter.capitalize()}',
            labels={'x': 'Clusters', 'y': 'Segments'}
        )

    # Count the frequency of each cluster
    cluster_counts = Counter(filtered_data)
    
    # Create a bar chart with Plotly Express
    figure = px.bar(
        x=list(cluster_counts.keys()),
        y=list(cluster_counts.values()),
        labels={'x': 'Clusters', 'y': 'Segments'},
        title=f'Segmentos por Clúster - {selected_filter.capitalize()}'
    )

    return figure
