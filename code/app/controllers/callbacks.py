import dash
from dash import html, dcc, callback_context
import zipfile
import shutil
import os
import io
import plotly.express as px
from collections import Counter
import time

from app.utils.config import UPLOAD_FOLDER
from app.utils.data_saveload import save_data, load_data
from app.utils.data_utils import list_files_in_folder
from app.views.layout.navbar import get_navbar
from app.views.layout.dataupload_page import get_page_dataupdate
from app.views.layout.experiment_page import get_page_experiment
from app.views.layout.map_page import get_page_map, get_map_image_as_html
from app.views.layout.select_page import get_page_select
from app.views.layout.TRACLUSmap_page import get_page_maptraclus, get_clusters_map
from app.views.layout.table_page import get_page_tables, get_table
from app.controllers.clustering import data_constructor

# -- Callbacks section --

def display_page(pathname):
    if pathname == '/':
        return get_page_select()
    elif pathname == '/new-experiment':
        return get_page_experiment()
    elif pathname == '/data-update':
        return get_page_dataupdate()
    elif pathname == '/map-page':
        return get_page_map()
    elif pathname == '/TRACLUS-map':
        return get_page_maptraclus(optics_on, hdbscan_on, dbscan_on, spect_on, aggl_on)
    elif pathname == '/estadisticas':
        return get_page_tables(optics_on, hdbscan_on, dbscan_on, spect_on, aggl_on)
    else:
        return get_page_select()

# -- Callbacks for navbar --

def update_navbar(pathname):
    return get_navbar(pathname)

# Callbacks for download data
def download_data(n_clicks):
    # Crear un archivo ZIP en memoria
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, mode="w") as zf:
        # Diccionarios para archivos de texto e imágenes
        images = {
            "map.png": html_map,
            "heatmap.png": html_heatmap
        }
        txt_files = {}

        # Crear archivos TXT en memoria y agregarlos al ZIP según las condiciones
        if optics_on:
            txt_files["tabla_optics.txt"] = tabla_optics.to_csv(index=False, sep='\t')
            images["traclus_map_optics.png"] = traclus_map_optics
            images["traclus_map_segments_optics.png"] = traclus_map_segments_optics
            images["traclus_map_cluster_optics.png"] = traclus_map_cluster_optics
        if hdbscan_on:
            txt_files["tabla_hdbscan.txt"] = tabla_hdbscan.to_csv(index=False, sep='\t')
            images["traclus_map_hdbscan.png"] = traclus_map_hdbscan
            images["traclus_map_segments_hdbscan.png"] = traclus_map_segments_hdbscan
            images["traclus_map_cluster_hdbscan.png"] = traclus_map_cluster_hdbscan
        if dbscan_on:
            txt_files["tabla_dbscan.txt"] = tabla_dbscan.to_csv(index=False, sep='\t')
            images["traclus_map_dbscan.png"] = traclus_map_dbscan
            images["traclus_map_segments_dbscan.png"] = traclus_map_segments_dbscan
            images["traclus_map_cluster_dbscan.png"] = traclus_map_cluster_dbscan
        if spect_on:
            txt_files["tabla_spectralclustering.txt"] = tabla_spect.to_csv(index=False, sep='\t')
            images["traclus_map_spectralclustering.png"] = traclus_map_spect
            images["traclus_map_segments_spectralclustering.png"] = traclus_map_segments_spect
            images["traclus_map_cluster_spectralclustering.png"] = traclus_map_cluster_spect
        if aggl_on:
            txt_files["tabla_agglomerativeclustering.txt"] = tabla_aggl.to_csv(index=False, sep='\t')
            images["traclus_map_agglomerativeclustering.png"] = traclus_map_aggl
            images["traclus_map_segments_agglomerativeclustering.png"] = traclus_map_segments_aggl
            images["traclus_map_cluster_agglomerativeclustering.png"] = traclus_map_cluster_aggl

        # Agregar los archivos TXT al ZIP
        for filename, txt_content in txt_files.items():
            zf.writestr(filename, txt_content)

        # Asegurarse de que cada imagen es un objeto BytesIO y agregar al ZIP
        for img_name, img_data in images.items():
            if not isinstance(img_data, io.BytesIO):  # Verificar si la imagen no es ya un BytesIO
                img_buffer = io.BytesIO(img_data)     # Convertir a BytesIO si es necesario
            else:
                img_buffer = img_data
            img_buffer.seek(0)  # Asegurarse de que el puntero esté al inicio del buffer
            zf.writestr(img_name, img_buffer.read())  # Agregar la imagen al ZIP

    # Coloca el puntero al inicio para leer el contenido
    zip_buffer.seek(0)

    # Enviar el archivo ZIP para descarga en el navegador
    return dcc.send_bytes(zip_buffer.getvalue(), "table.zip")

# -- Callbacks for select page --

def navigate_experiment_page( n_clicks_new):
    if n_clicks_new > 0:
        return '/new-experiment'
    return '/'

def display_files_in_selected_folder(folder_name, n_clicks_previous, is_modal_open):
    if n_clicks_previous > 0:
        if folder_name is None:
            return dash.no_update, not is_modal_open

        # Obtener los archivos dentro de la carpeta seleccionada
        files = list_files_in_folder(folder_name)

        load = load_data(files, folder_name)

        global optics_on, dbscan_on, hdbscan_on, aggl_on, spect_on
        global gdf, html_map, html_heatmap
        global traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics
        global traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan
        global traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan
        global traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect
        global traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl
        
        optics_on, dbscan_on, hdbscan_on, aggl_on, spect_on, \
        gdf, html_map, html_heatmap, \
        traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics, \
        traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan, \
        traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan, \
        traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect, \
        traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl = load

        if optics_on or dbscan_on or hdbscan_on or aggl_on or spect_on:
            return '/map-page', is_modal_open
    
    return dash.no_update, is_modal_open

def toggle_modal(n_delete, n_cancel, n_confirm, is_open, folder_name):
    if folder_name is not None and (n_delete or n_cancel or n_confirm):
        return not is_open
    return is_open

def delete_experiment(n_clicks, folder_name):
    if n_clicks and folder_name:
        folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            return "/"
    return dash.no_update

# -- Callbacks for experiment page --

def navigate_to_page_dataupdate(n_clicks_data, checkoptics, optics_metric_value, optics_algorithm_value, optics_eps_value, optics_sample_value, checkdbscan, 
                                dbscan_metric_value, dbscan_algorithm_value, dbscan_eps_value, dbscan_sample_value, checkhdbscan, hdbscan_metric_value, 
                                hdbscan_algorithm_value, hdbscan_sample_value, checkagglomerativeclustering, aggl_metric_value, aggl_linkage_value, 
                                aggl_n_clusters_value, checkspectralclustering, spect_affinity_value, spect_assign_labels_value, spect_n_clusters_value,
                                is_open):
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

        if checkoptics and optics_metric_value and optics_algorithm_value and optics_eps_value and optics_sample_value:
            optics_on = True
            optics_metric = optics_metric_value
            optics_algorithm = optics_algorithm_value
            optics_eps = optics_eps_value
            optics_sample = optics_sample_value
        if checkdbscan and dbscan_metric_value and dbscan_algorithm_value and dbscan_eps_value and dbscan_sample_value:
            dbscan_on = True
            dbscan_metric = dbscan_metric_value
            dbscan_algorithm = dbscan_algorithm_value
            dbscan_eps = dbscan_eps_value
            dbscan_sample = dbscan_sample_value
        if checkhdbscan and hdbscan_metric_value and hdbscan_algorithm_value and hdbscan_sample_value:
            hdbscan_on = True
            hdbscan_metric = hdbscan_metric_value
            hdbscan_algorithm = hdbscan_algorithm_value
            hdbscan_sample = hdbscan_sample_value
        if checkagglomerativeclustering and aggl_metric_value and aggl_linkage_value and aggl_n_clusters_value:
            aggl_on = True
            aggl_metric = aggl_metric_value
            aggl_linkage = aggl_linkage_value
            aggl_n_clusters = aggl_n_clusters_value
        if checkspectralclustering and spect_affinity_value and spect_assign_labels_value and spect_n_clusters_value:
            spect_on = True
            spect_affinity = spect_affinity_value
            spect_assign_labels = spect_assign_labels_value
            spect_n_clusters = spect_n_clusters_value

        if optics_on or dbscan_on or hdbscan_on or aggl_on or spect_on:
            return '/data-update', is_open
        return dash.no_update, not is_open

    return '/new-experiment', is_open

def toggle_rowo_controls(selector_value_o):
        is_enabled = 'on' in selector_value_o
        return not is_enabled, not is_enabled, not is_enabled, not is_enabled

def toggle_rowd_controls(selector_value_d):
        is_enabled = 'on' in selector_value_d
        return not is_enabled, not is_enabled, not is_enabled, not is_enabled

def toggle_rowh_controls(selector_value_h):
        is_enabled = 'on' in selector_value_h
        return not is_enabled, not is_enabled, not is_enabled

def toggle_rowa_controls(selector_value_a):
        is_enabled = 'on' in selector_value_a
        return not is_enabled, not is_enabled, not is_enabled

def toggle_rows_controls(selector_value_s):
        is_enabled = 'on' in selector_value_s
        return not is_enabled, not is_enabled, not is_enabled

# -- Callbacks for data upload page --

def process_csv_from_url(n_clicks_upload, data, nrows, folder_name):
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

        global gdf, tray, html_map, html_heatmap
        global traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics
        global traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan
        global traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan
        global traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect
        global traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl
        global error_message

        gdf, tray, html_map, html_heatmap, \
        traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics, \
        traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan, \
        traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan, \
        traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect, \
        traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl, \
        error_message = result

        # Manejar el mensaje de error
        if error_message:
            return dash.no_update, html.Div([error_message])
        
        # Guardar los resultados en una carpeta
        save_data(folder_name, gdf, html_map, html_heatmap, 
            traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics, 
            traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan, 
            traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan, 
            traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect, 
            traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl)

        return '/map-page', html.Div(['Procesamiento exitoso.'])

    # Si no se ha hecho clic en el botón, no realizar cambios
    return dash.no_update, dash.no_update

# -- Callbacks for map page --

def update_map(*args):            
    map_image = get_map_image_as_html(html_map, html_heatmap)
    return [map_image]

# -- Callbacks for TRACLUS map page --

def display_clusters_1(*args):
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
    ctx = callback_context

    if not ctx.triggered:
        if optics_on:
            return get_clusters_map(traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics)
        elif hdbscan_on:
            return get_clusters_map(traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan)
        elif dbscan_on:
            return get_clusters_map(traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan)
        elif spect_on:
            return get_clusters_map(traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect)
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
    ctx = callback_context

    if not ctx.triggered:
        if optics_on:
            return get_table(tabla_optics)
        elif hdbscan_on:
            return get_table(tabla_hdbscan)
        elif dbscan_on:
            return get_table(tabla_dbscan)
        elif spect_on:
            return get_table(tabla_spect)
        elif aggl_on:
            return get_table(tabla_aggl)
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'table-1':
            return get_table(tabla_optics)
        elif button_id == 'table-2':
            return get_table(tabla_hdbscan)
        elif button_id == 'table-3':
            return get_table(tabla_dbscan)
        elif button_id == 'table-4':
            return get_table(tabla_spect)
        elif button_id == 'table-5':
            return get_table(tabla_aggl)
        
def update_graph(selected_filter):
    # Si no hay una selección, devolver un gráfico vacío
    if not selected_filter:
        return px.bar(
            title='Gráfico vacío',
            labels={'x': 'Clusters', 'y': 'Segments'}
        )

    # Filtrar los datos según la selección del usuario
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
        # Si el filtro seleccionado no tiene datos o está deshabilitado
        return px.bar(
            title=f'No hay datos para {selected_filter.capitalize()}',
            labels={'x': 'Clusters', 'y': 'Segments'}
        )

    # Contar la frecuencia de cada clúster
    cluster_counts = Counter(filtered_data)
    
    # Crear el gráfico de barras con Plotly Express
    figure = px.bar(
        x=list(cluster_counts.keys()),
        y=list(cluster_counts.values()),
        labels={'x': 'Clusters', 'y': 'Segments'},
        title=f'Segmentos por Clúster - {selected_filter.capitalize()}'
    )

    return figure