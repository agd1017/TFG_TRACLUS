import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback_context
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import zipfile
import shutil
import os
import io

from utils.config import UPLOAD_FOLDER, TRAIN_DATA
from utils.data_utils import list_files_in_folder, save_html_or_binary
from views.layout.dataupload_page import get_page_dataupdate
from views.layout.experiment_page import get_page_experiment
from views.layout.map_page import get_page_map, get_map_image_as_html
from views.layout.select_page import get_page_select
from views.layout.TRACLUSmap_page import get_page_maptraclus, get_clusters_map
from views.layout.table_page import get_page_tables, get_table
from controllers.clustering import data_constructor

# Callbacks
def register_upload_callbacks(app):
    @app.callback(
        Output('navbar-container', 'children'),
        [Input('url', 'pathname')]
    )
    def update_navbar(pathname):
        disabled = pathname in ['/', '/new-experiment', '/data-update']

        # Generar la barra de navegación dinámicamente con los botones habilitados o deshabilitados
        navbar = dbc.Navbar(
            dbc.Container(children=[
                dbc.NavItem(dbc.NavLink("TRACLUS", href="/", className="navbar-text-title", disabled=(pathname == '/'))),
                dbc.NavItem(dbc.NavLink("Mapa de trayectorias", href="/map-page", className="navbar-text", disabled=disabled)),
                dbc.NavItem(dbc.NavLink("Comparacion de algoritmos", href="/TRACLUS-map", className="navbar-text", disabled=disabled)),
                dbc.NavItem(dbc.NavLink("Estadísticas", href="/estadisticas", className="navbar-text", disabled=disabled)),
                # Botón de descarga de datos
                dbc.NavItem([
                    dbc.Button("Descargar Datos", id="btn-download-txt", className="navbar-text", disabled=disabled),
                    dcc.Download(id="download-text"),
                ])
            ]),
            color="success",
            className="header-navbar"
        )
        
        return navbar

    @app.callback(
        Output('page-content', 'children'),
        [Input('url', 'pathname')]
    )
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

    # Callbacks for select page

    def read_html_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'rb') as f:
                return f.read()  # Devuelve como bytes
            
    def convert_to_dataframe(file_path):
        # Lee el archivo CSV y lo convierte a un DataFrame
        return pd.read_csv(file_path)

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        Input('new-exp-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def navigate_to_page( n_clicks_new):
        if n_clicks_new > 0:
            return '/new-experiment'
        return '/'

    def load_data(files, folder_name):
        global gdf, tray, html_map, html_heatmap
        global traclus_map_optics, traclus_map_cluster_optics, traclus_map_segments_optics, tabla_optics, graph_optics
        global traclus_map_hdbscan, traclus_map_cluster_hdbscan, traclus_map_segments_hdbscan, tabla_hdbscan, graph_hdbscan
        global traclus_map_dbscan, traclus_map_cluster_dbscan, traclus_map_segments_dbscan, tabla_dbscan, graph_dbscan
        global traclus_map_spect, traclus_map_cluster_spect, traclus_map_segments_spect, tabla_spect, graph_spect
        global traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl, tabla_aggl, graph_aggl
        global error_message
        global optics_on, dbscan_on, hdbscan_on, aggl_on, spect_on 

        # Reiniciar las variables globales
        optics_on = dbscan_on = hdbscan_on = aggl_on = spect_on  = False
        gdf = tray = html_map = html_heatmap = None
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

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        #Output('file-list-container', 'children'),
        [Input('experiment-dropdown', 'value'),
        Input('load-exp-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def display_files_in_selected_folder(folder_name, n_clicks_previous):
        if n_clicks_previous > 0:
            if folder_name is None:
                return dash.no_update #, html.Div(["Selecciona una carpeta para ver los archivos."])

            # Obtener los archivos dentro de la carpeta seleccionada
            files = list_files_in_folder(folder_name)
            
            load_data(files, folder_name)

            if optics_on or dbscan_on or hdbscan_on or aggl_on or spect_on:
                return '/map-page' #, {}
        
        return dash.no_update #, dash.no_update
    
    @app.callback(
        Output("delete-modal", "is_open"),
        [Input("delete-exp-button", "n_clicks"),
        Input("cancel-delete-button", "n_clicks"),
        Input("confirm-delete-button", "n_clicks")],
        [State("delete-modal", "is_open"),
        State("experiment-dropdown", "value")],
        prevent_initial_call=True
    )
    def toggle_modal(n_delete, n_cancel, n_confirm, is_open, folder_name):
        if folder_name is not None:
            if n_delete or n_cancel or n_confirm:
                return not is_open
        return is_open


    @app.callback(
        Output("url", "pathname"),
        [Input("confirm-delete-button", "n_clicks")],
        [State("experiment-dropdown", "value")],
        prevent_initial_call=True
    )
    def delete_experiment(n_clicks, folder_name):
        if n_clicks and folder_name:
            folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                return "/"
        return dash.no_update
    

    # Callbacks for experiment page

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        Input('execute-button', 'n_clicks'),
        [State('selector-optics', 'value'),
        State('dropdown-optics-metric', 'value'),
        State('dropdown-optics-algorithm', 'value'),
        State('input-optics-eps', 'value'),
        State('input-optics-sample', 'value'),
        State('selector-dbscan', 'value'),
        State('dropdown-dbscan-metric', 'value'),
        State('dropdown-dbscan-algorithm', 'value'),
        State('input-dbscan-eps', 'value'),
        State('input-dbscan-sample', 'value'),
        State('selector-hdbscan', 'value'),
        State('dropdown-hdbscan-metric', 'value'),
        State('dropdown-hdbscan-algorithm', 'value'),
        State('input-hdbscan-sample', 'value'),
        State('selector-agglomerativeclustering', 'value'),
        State('dropdown-agglomerativeclustering-metric', 'value'),
        State('dropdown-agglomerativeclustering-linkage', 'value'),
        State('input-agglomerativeclustering-n_clusters', 'value'),
        State('selector-spectralclustering', 'value'),
        State('dropdown-spectralclustering-affinity', 'value'),
        State('dropdown-spectralclustering-assign_labels', 'value'),
        State('input-spectralclustering-n_clusters', 'value')],
        prevent_initial_call=True
    )
    def navigate_to_page_dataupdate(n_clicks_data, checkoptics, optics_metric_value, optics_algorithm_value, optics_eps_value, optics_sample_value, checkdbscan, 
                                    dbscan_metric_value, dbscan_algorithm_value, dbscan_eps_value, dbscan_sample_value, checkhdbscan, hdbscan_metric_value, 
                                    hdbscan_algorithm_value, hdbscan_sample_value, checkagglomerativeclustering, aggl_metric_value, aggl_linkage_value, 
                                    aggl_n_clusters_value, checkspectralclustering, spect_affinity_value, spect_assign_labels_value, spect_n_clusters_value):
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
                return '/data-update'
        return '/new-experiment'

    # Callbacks for disable/enable selectors
    @app.callback(
        [Output('dropdown-optics-metric', 'disabled'),
        Output('dropdown-optics-algorithm', 'disabled'),
        Output('input-optics-eps', 'disabled'),
        Output('input-optics-sample', 'disabled')],
        [Input('selector-optics', 'value')]
    )
    def toggle_rowo_controls(selector_value_o):
            is_enabled = 'on' in selector_value_o
            return not is_enabled, not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output('dropdown-dbscan-metric', 'disabled'),
        Output('dropdown-dbscan-algorithm', 'disabled'),
        Output('input-dbscan-eps', 'disabled'),
        Output('input-dbscan-sample', 'disabled')],
        [Input('selector-dbscan', 'value')]
    )
    def toggle_rowd_controls(selector_value_d):
            is_enabled = 'on' in selector_value_d
            return not is_enabled, not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output('dropdown-hdbscan-metric', 'disabled'),
        Output('dropdown-hdbscan-algorithm', 'disabled'),
        Output('input-hdbscan-sample', 'disabled')],
        [Input('selector-hdbscan', 'value')]
    )
    def toggle_rowh_controls(selector_value_h):
            is_enabled = 'on' in selector_value_h
            return not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output('dropdown-agglomerativeclustering-metric', 'disabled'),
        Output('dropdown-agglomerativeclustering-linkage', 'disabled'),
        Output('input-agglomerativeclustering-n_clusters', 'disabled')],
        [Input('selector-agglomerativeclustering', 'value')]
    )
    def toggle_rowa_controls(selector_value_a):
            is_enabled = 'on' in selector_value_a
            return not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output('dropdown-spectralclustering-affinity', 'disabled'),
        Output('dropdown-spectralclustering-assign_labels', 'disabled'),
        Output('input-spectralclustering-n_clusters', 'disabled')],
        [Input('selector-spectralclustering', 'value')]
    )
    def toggle_rows_controls(selector_value_s):
            is_enabled = 'on' in selector_value_s
            return not is_enabled, not is_enabled, not is_enabled

    # Callbacks for data upload page

    def save_data(folder_name):
        # Crear la carpeta en la ubicación deseada
        folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
        # Si la carpeta existe, eliminarla
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Elimina la carpeta y todo su contenido
        
        # Crear la carpeta nuevamente
        os.makedirs(folder_path)

        # Guardar los datos en la carpeta creada:
        # 1. Guardar gdf como archivo GeoJSON
        if gdf is not None:
            gdf.to_file(os.path.join(folder_path, "resultado_gdf.geojson"), driver='GeoJSON')

        # 2. Guardar los mapas HTML, si existen
        save_html_or_binary(os.path.join(folder_path, "html_map.html"), html_map if html_map is not None else "")
        save_html_or_binary(os.path.join(folder_path, "html_heatmap.html"), html_heatmap if html_heatmap is not None else "")

        # 3. Guardar cada uno de los mapas de TRACLUS en formato HTML si no son None
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

        # 4. Guardar los mapas con segmentos en archivos HTML si no son None
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

        # 5. Guardar los mapas con clusters en archivos HTML si no son None
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

        # 6. Guardar las tablas generadas por cada algoritmo en CSV si no son None
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

        # 7. Guardar los gráficos como CSV
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


    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        Output('output-container', 'children', allow_duplicate=True),
        Input('default-config-button', 'n_clicks'),
        Input('input-name', 'value'), 
        prevent_initial_call=True
    )

    def upload_output_predeter(n_clicks_upload, folder_name):
        if n_clicks_upload is not None and n_clicks_upload > 0:
            if not folder_name:
                return dash.no_update, html.Div(["Por favor, introduce un nombre para el experimento."])

            data = TRAIN_DATA
            nrows = 5

            result = data_constructor(data, nrows, optics_on, optics_metric, optics_algorithm, optics_eps, optics_sample, 
                                    dbscan_on, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample, 
                                    hdbscan_on, hdbscan_metric, hdbscan_algorithm, hdbscan_sample, 
                                    aggl_on, aggl_metric, aggl_linkage, aggl_n_clusters, 
                                    spect_on, spect_affinity, spect_assign_labels, spect_n_clusters)

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
            save_data(folder_name)

            return '/map-page', html.Div(['Procesamiento exitoso.'])

        # Si no se ha hecho clic en el botón, no realizar cambios
        return dash.no_update, dash.no_update

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        Output('output-container', 'children', allow_duplicate=True),
        Input('process-url-button', 'n_clicks'),
        Input('upload-data', 'contents'),
        # State('input-url', 'value'),
        Input('nrows-input', 'value'),
        Input('input-name', 'value'), 
        prevent_initial_call=True
    )

    def process_csv_from_url(n_clicks_upload, data, nrows, folder_name):
        if n_clicks_upload is not None and n_clicks_upload > 0:
            if not folder_name:
                return dash.no_update, html.Div(["Por favor, introduce un nombre para el experimento."])
            if not data:
                return dash.no_update, html.Div(["No se ha introducido ningún enlace."])
            if not nrows:
                return dash.no_update, html.Div(["No se ha introducido el número de filas."])
            
            result = data_constructor(data, nrows, optics_on, optics_metric, optics_algorithm, optics_eps, optics_sample, 
                                    dbscan_on, dbscan_metric, dbscan_algorithm, dbscan_eps, dbscan_sample, 
                                    hdbscan_on, hdbscan_metric, hdbscan_algorithm, hdbscan_sample, 
                                    aggl_on, aggl_metric, aggl_linkage, aggl_n_clusters, 
                                    spect_on, spect_affinity, spect_assign_labels, spect_n_clusters)

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
            save_data(folder_name)

            return '/map-page', html.Div(['Procesamiento exitoso.'])

        # Si no se ha hecho clic en el botón, no realizar cambios
        return dash.no_update, dash.no_update

    # Callbacks for map page
        
    @app.callback(
        Output('map-container', 'children'),
        [Input('option-1-1', 'n_clicks'),
        Input('option-1-2', 'n_clicks'),
        Input('option-1-3', 'n_clicks')]
    )

    def update_map(*args):            
        map_image = get_map_image_as_html(html_map, html_heatmap)
        return [map_image]

    # Callbacks for TRACLUS map page

    @app.callback(
        Output('map-clusters-1', 'children'),
        [Input('item-1-1', 'n_clicks'),
        Input('item-1-2', 'n_clicks'),
        Input('item-1-3', 'n_clicks'),
        Input('item-1-4', 'n_clicks'),
        Input('item-1-5', 'n_clicks')]
    )
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
                return get_clusters_map(traclus_map_spect, traclus_map_cluster_spect,traclus_map_segments_spect)
            elif button_id == 'item-1-5':
                return get_clusters_map(traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl)
            
    @app.callback(
        Output('map-clusters-2', 'children'),
        [Input('item-2-1', 'n_clicks'),
        Input('item-2-2', 'n_clicks'),
        Input('item-2-3', 'n_clicks'),
        Input('item-2-4', 'n_clicks'),
        Input('item-2-5', 'n_clicks')]
    )

    def display_clusters_2(*args):
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
                return get_clusters_map(traclus_map_spect, traclus_map_cluster_spect,traclus_map_segments_spect)
            elif button_id == 'item-1-5':
                return get_clusters_map(traclus_map_aggl, traclus_map_cluster_aggl, traclus_map_segments_aggl)

    # Callbacks for tables page

    @app.callback(
        Output('table-container', 'children'),
        [Input('table-1', 'n_clicks'),
        Input('table-2', 'n_clicks'),
        Input('table-3', 'n_clicks'),
        Input('table-4', 'n_clicks'),
        Input('table-5', 'n_clicks')],
        prevent_initial_call=True
    )

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
            
    # Callback para actualizar el gráfico
    import plotly.express as px
    from collections import Counter
            
    @app.callback(
        Output('cluster-bar-chart', 'figure'),
        Input('cluster-selector', 'value')
    )
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
    
    # Callbacks for download data

    @app.callback(
        Output("download-text", "data"),
        Input("btn-download-txt", "n_clicks"),
        prevent_initial_call=True,
    )
    def func(n_clicks):
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