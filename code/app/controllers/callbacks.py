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
from views.layout.dataupload_page import get_page_dataUpdate
from views.layout.experiment_page import get_page_experiment
from views.layout.map_page import get_page_map, get_map_image_as_html
from views.layout.select_page import get_page_select
from views.layout.TRACLUSmap_page import get_page_mapTRACLUS, get_clusters_map
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
            return get_page_dataUpdate()
        elif pathname == '/map-page':
            return get_page_map()
        elif pathname == '/TRACLUS-map':
            return get_page_mapTRACLUS(OPTICS_ON, HDBSCAN_ON, DBSCAN_ON, Spect_ON, Aggl_ON)
        elif pathname == '/estadisticas':
            return get_page_tables(OPTICS_ON, HDBSCAN_ON, DBSCAN_ON, Spect_ON, Aggl_ON)
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
        global gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
            TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN,TRACLUS_map_cluster_OPTICS, \
            TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_cluster_DBSCAN, \
            TRACLUS_map_cluster_Spect, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_HDBSCAN, \
            TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_Spect, \
            TRACLUS_map_df_Spect, TRACLUS_map_Aggl, TRACLUS_map_df_Aggl, \
            tabla_OPTICS, tabla_HDBSCAN, tabla_DBSCAN, tabla_Spect, tabla_Aggl, \
            graph_OPTICS, graph_HDBSCAN, graph_DBSCAN, graph_Spect, graph_Aggl
        global OPTICS_ON, DBSCAN_ON, HDBSCAN_ON, Aggl_ON, Spect_ON 

        # Reiniciar las variables globales
        OPTICS_ON, DBSCAN_ON, HDBSCAN_ON, Aggl_ON, Spect_ON  = False, False, False, False, False
        gdf = tray = html_map = html_heatmap = TRACLUS_map_OPTICS = None
        TRACLUS_map_df_OPTICS = TRACLUS_map_HDBSCAN = TRACLUS_map_df_HDBSCAN = None
        TRACLUS_map_DBSCAN = TRACLUS_map_df_DBSCAN = TRACLUS_map_Spect = None
        TRACLUS_map_df_Spect = TRACLUS_map_Aggl = None
        TRACLUS_map_df_Aggl = tabla_OPTICS = tabla_HDBSCAN = None
        tabla_DBSCAN = tabla_Spect = tabla_Aggl = None
        graph_OPTICS = graph_HDBSCAN = graph_DBSCAN = graph_Spect = graph_Aggl = None
        TRACLUS_map_cluster_Aggl = TRACLUS_map_cluster_DBSCAN = TRACLUS_map_cluster_HDBSCAN = None
        TRACLUS_map_cluster_OPTICS = TRACLUS_map_cluster_Spect = None

        for file in files:
                file_path = os.path.join(UPLOAD_FOLDER, folder_name, file)

                if file == "resultado_gdf.geojson":
                    gdf = gpd.read_file(file_path)
                elif file == "html_map.html":
                    html_map = read_html_file(file_path)
                elif file == "html_heatmap.html":
                    html_heatmap = read_html_file(file_path)
                elif file == "TRACLUS_map_OPTICS.html":
                    OPTICS_ON = True
                    TRACLUS_map_OPTICS = read_html_file(file_path)
                elif file == "TRACLUS_map_df_OPTICS.html":
                    TRACLUS_map_df_OPTICS = read_html_file(file_path)
                elif file == "TRACLUS_map_HDBSCAN.html":
                    HDBSCAN_ON = True
                    TRACLUS_map_HDBSCAN = read_html_file(file_path)
                elif file == "TRACLUS_map_df_HDBSCAN.html":
                    TRACLUS_map_df_HDBSCAN = read_html_file(file_path)
                elif file == "TRACLUS_map_DBSCAN.html":
                    DBSCAN_ON = True
                    TRACLUS_map_DBSCAN = read_html_file(file_path)
                elif file == "TRACLUS_map_df_DBSCAN.html":
                    TRACLUS_map_df_DBSCAN = read_html_file(file_path)
                elif file == "TRACLUS_map_SpectralClustering.html":
                    Spect_ON = True
                    TRACLUS_map_SpectralClustering = read_html_file(file_path)
                elif file == "TRACLUS_map_df_SpectralClustering.html":
                    TRACLUS_map_df_SpectralClustering = read_html_file(file_path)
                elif file == "TRACLUS_map_AgglomerativeClustering.html":
                    Aggl_ON = True
                    TRACLUS_map_AgglomerativeClustering = read_html_file(file_path)
                elif file == "TRACLUS_map_df_AgglomerativeClustering.html":
                    TRACLUS_map_df_AgglomerativeClustering = read_html_file(file_path)
                elif file == "tabla_OPTICS.csv":
                    tabla_OPTICS = convert_to_dataframe(file_path)
                elif file == "tabla_HDBSCAN.csv":
                    tabla_HDBSCAN = convert_to_dataframe(file_path)
                elif file == "tabla_DBSCAN.csv":
                    tabla_DBSCAN = convert_to_dataframe(file_path)
                elif file == "tabla_SpectralClustering.csv":
                    tabla_SpectralClustering = convert_to_dataframe(file_path)
                elif file == "tabla_AgglomerativeClustering.csv":
                    tabla_AgglomerativeClustering = convert_to_dataframe(file_path)
                elif file == "graph_OPTICS.csv":
                    graph_OPTICS = pd.read_csv(file_path)['Data'].tolist()
                elif file == "graph_HDBSCAN.csv":
                    graph_HDBSCAN = pd.read_csv(file_path)['Data'].tolist()
                elif file == "graph_DBSCAN.csv":
                    graph_DBSCAN = pd.read_csv(file_path)['Data'].tolist()
                elif file == "graph_SpectralClustering.csv":
                    graph_Spect = pd.read_csv(file_path)['Data'].tolist()
                elif file == "graph_AgglomerativeClustering.csv":
                    graph_Aggl = pd.read_csv(file_path)['Data'].tolist()

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        #Output('file-list-container', 'children'),
        [Input('experiment-dropdown', 'value'),
        Input('previous-exp-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def display_files_in_selected_folder(folder_name, n_clicks_previous):
        if n_clicks_previous > 0:
            if folder_name is None:
                return dash.no_update #, html.Div(["Selecciona una carpeta para ver los archivos."])

            # Obtener los archivos dentro de la carpeta seleccionada
            files = list_files_in_folder(folder_name)
            
            load_data(files, folder_name)

            if OPTICS_ON or DBSCAN_ON or HDBSCAN_ON or Aggl_ON or Spect_ON:
                return '/map-page' #, {}
        
        return dash.no_update #, dash.no_update

    # Callbacks for experiment page

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        Input('execute-button', 'n_clicks'),
        [State('selector-OPTICS', 'value'),
        State('dropdown-OPTICS-metric', 'value'),
        State('dropdown-OPTICS-algorithm', 'value'),
        State('input-OPTICS-eps', 'value'),
        State('input-OPTICS-sample', 'value'),
        State('selector-DBSCAN', 'value'),
        State('dropdown-DBSCAN-metric', 'value'),
        State('dropdown-DBSCAN-algorithm', 'value'),
        State('input-DBSCAN-eps', 'value'),
        State('input-DBSCAN-sample', 'value'),
        State('selector-HDBSCAN', 'value'),
        State('dropdown-HDBSCAN-metric', 'value'),
        State('dropdown-HDBSCAN-algorithm', 'value'),
        State('input-HDBSCAN-sample', 'value'),
        State('selector-AgglomerativeClustering', 'value'),
        State('dropdown-AgglomerativeClustering-metric', 'value'),
        State('dropdown-AgglomerativeClustering-linkage', 'value'),
        State('input-AgglomerativeClustering-n_clusters', 'value'),
        State('selector-SpectralClustering', 'value'),
        State('dropdown-SpectralClustering-affinity', 'value'),
        State('dropdown-SpectralClustering-assign_labels', 'value'),
        State('input-SpectralClustering-n_clusters', 'value')],
        prevent_initial_call=True
    )
    def navigate_to_page_dataupdate(n_clicks_data, checkOptics, OPTICS_metric_value, OPTICS_algorithm_value, OPTICS_eps_value, OPTICS_sample_value, checkDBSCAN, 
                                    DBSCAN_metric_value, DBSCAN_algorithm_value, DBSCAN_eps_value, DBSCAN_sample_value, checkHDBSCAN, HDBSCAN_metric_value, 
                                    HDBSCAN_algorithm_value, HDBSCAN_sample_value, checkAgglomerativeClustering, Aggl_metric_value, Aggl_linkage_value, 
                                    Aggl_n_clusters_value, checkSpectralClustering, Spect_affinity_value, Spect_assign_labels_value, Spect_n_clusters_value):
        if n_clicks_data is not None and n_clicks_data > 0:
            global OPTICS_ON, DBSCAN_ON, HDBSCAN_ON, Aggl_ON, Spect_ON 
            OPTICS_ON, DBSCAN_ON, HDBSCAN_ON, Aggl_ON, Spect_ON  = False, False, False, False, False
            
            global OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample
            OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample = None, None, None, None
            global DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample
            DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample = None, None, None, None
            global HDBSCAN_metric, HDBSCAN_algorithm, HDBSCAN_sample
            HDBSCAN_metric, HDBSCAN_algorithm, HDBSCAN_sample = None, None, None
            global Aggl_metric, Aggl_linkage, Aggl_n_clusters
            Aggl_metric, Aggl_linkage, Aggl_n_clusters = None, None, None
            global Spect_affinity, Spect_assign_labels, Spect_n_clusters
            Spect_affinity, Spect_assign_labels, Spect_n_clusters = None, None, None

            if checkOptics and OPTICS_metric_value and OPTICS_algorithm_value and OPTICS_eps_value and OPTICS_sample_value:
                OPTICS_ON = True
                OPTICS_metric = OPTICS_metric_value
                OPTICS_algorithm = OPTICS_algorithm_value
                OPTICS_eps = OPTICS_eps_value
                OPTICS_sample = OPTICS_sample_value
            if checkDBSCAN and DBSCAN_metric_value and DBSCAN_algorithm_value and DBSCAN_eps_value and DBSCAN_sample_value:
                DBSCAN_ON = True
                DBSCAN_metric = DBSCAN_metric_value
                DBSCAN_algorithm = DBSCAN_algorithm_value
                DBSCAN_eps = DBSCAN_eps_value
                DBSCAN_sample = DBSCAN_sample_value
            if checkHDBSCAN and HDBSCAN_metric_value and HDBSCAN_algorithm_value and HDBSCAN_sample_value:
                HDBSCAN_ON = True
                HDBSCAN_metric = HDBSCAN_metric_value
                HDBSCAN_algorithm = HDBSCAN_algorithm_value
                HDBSCAN_sample = HDBSCAN_sample_value
            if checkAgglomerativeClustering and Aggl_metric_value and Aggl_linkage_value and Aggl_n_clusters_value:
                Aggl_ON = True
                Aggl_metric = Aggl_metric_value
                Aggl_linkage = Aggl_linkage_value
                Aggl_n_clusters = Aggl_n_clusters_value
            if checkSpectralClustering and Spect_affinity_value and Spect_assign_labels_value and Spect_n_clusters_value:
                Spect_ON = True
                Spect_affinity = Spect_affinity_value
                Spect_assign_labels = Spect_assign_labels_value
                Spect_n_clusters = Spect_n_clusters_value

            #print(f"{OPTICS_metric_value}, {OPTICS_algorithm_value}, {OPTICS_eps_value}, {OPTICS_sample_value}")

            if OPTICS_ON or DBSCAN_ON or HDBSCAN_ON or Aggl_ON or Spect_ON:
                return '/data-update'
        return '/new-experiment'

    # Callbacks for disable/enable selectors
    @app.callback(
        [Output(f'dropdown-OPTICS-metric', 'disabled'),
        Output(f'dropdown-OPTICS-algorithm', 'disabled'),
        Output(f'input-OPTICS-eps', 'disabled'),
        Output(f'input-OPTICS-sample', 'disabled')],
        [Input(f'selector-OPTICS', 'value')]
    )
    def toggle_rowO_controls(selector_value_O):
            is_enabled = 'on' in selector_value_O
            return not is_enabled, not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output(f'dropdown-DBSCAN-metric', 'disabled'),
        Output(f'dropdown-DBSCAN-algorithm', 'disabled'),
        Output(f'input-DBSCAN-eps', 'disabled'),
        Output(f'input-DBSCAN-sample', 'disabled')],
        [Input(f'selector-DBSCAN', 'value')]
    )
    def toggle_rowD_controls(selector_value_D):
            is_enabled = 'on' in selector_value_D
            return not is_enabled, not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output(f'dropdown-HDBSCAN-metric', 'disabled'),
        Output(f'dropdown-HDBSCAN-algorithm', 'disabled'),
        Output(f'input-HDBSCAN-sample', 'disabled')],
        [Input(f'selector-HDBSCAN', 'value')]
    )
    def toggle_rowH_controls(selector_value_H):
            is_enabled = 'on' in selector_value_H
            return not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output(f'dropdown-AgglomerativeClustering-metric', 'disabled'),
        Output(f'dropdown-AgglomerativeClustering-linkage', 'disabled'),
        Output(f'input-AgglomerativeClustering-n_clusters', 'disabled')],
        [Input(f'selector-AgglomerativeClustering', 'value')]
    )
    def toggle_rowA_controls(selector_value_A):
            is_enabled = 'on' in selector_value_A
            return not is_enabled, not is_enabled, not is_enabled

    @app.callback(
        [Output(f'dropdown-SpectralClustering-affinity', 'disabled'),
        Output(f'dropdown-SpectralClustering-assign_labels', 'disabled'),
        Output(f'input-SpectralClustering-n_clusters', 'disabled')],
        [Input(f'selector-SpectralClustering', 'value')]
    )
    def toggle_rowS_controls(selector_value_S):
            is_enabled = 'on' in selector_value_S
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
        if TRACLUS_map_OPTICS is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_OPTICS.html"), TRACLUS_map_OPTICS)

        if TRACLUS_map_HDBSCAN is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_HDBSCAN.html"), TRACLUS_map_HDBSCAN)

        if TRACLUS_map_DBSCAN is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_DBSCAN.html"), TRACLUS_map_DBSCAN)

        if TRACLUS_map_Spect is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_SpectralClustering.html"), TRACLUS_map_Spect)

        if TRACLUS_map_Aggl is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_AgglomerativeClustering.html"), TRACLUS_map_Aggl)

        # 4. Guardar los mapas con DataFrames en archivos HTML si no son None
        if TRACLUS_map_df_OPTICS is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_df_OPTICS.html"), TRACLUS_map_df_OPTICS)

        if TRACLUS_map_df_HDBSCAN is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_df_HDBSCAN.html"), TRACLUS_map_df_HDBSCAN)

        if TRACLUS_map_df_DBSCAN is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_df_DBSCAN.html"), TRACLUS_map_df_DBSCAN)

        if TRACLUS_map_df_Spect is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_df_SpectralClustering.html"), TRACLUS_map_df_Spect)

        if TRACLUS_map_df_Aggl is not None:
            save_html_or_binary(os.path.join(folder_path, "TRACLUS_map_df_AgglomerativeClustering.html"), TRACLUS_map_df_Aggl)

        # 5. Guardar las tablas generadas por cada algoritmo en CSV si no son None
        if tabla_OPTICS is not None:
            tabla_OPTICS.to_csv(os.path.join(folder_path, "tabla_OPTICS.csv"), index=False)

        if tabla_HDBSCAN is not None:
            tabla_HDBSCAN.to_csv(os.path.join(folder_path, "tabla_HDBSCAN.csv"), index=False)

        if tabla_DBSCAN is not None:
            tabla_DBSCAN.to_csv(os.path.join(folder_path, "tabla_DBSCAN.csv"), index=False)

        if tabla_Spect is not None:
            tabla_Spect.to_csv(os.path.join(folder_path, "tabla_SpectralClustering.csv"), index=False)

        if tabla_Aggl is not None:
            tabla_Aggl.to_csv(os.path.join(folder_path, "tabla_AgglomerativeClustering.csv"), index=False)

        # 6. Guardar los gráficos como CSV
        if graph_OPTICS is not None:
            pd.DataFrame({'Data': graph_OPTICS}).to_csv(os.path.join(folder_path, "graph_OPTICS.csv"), index=False)

        if graph_HDBSCAN is not None:
            pd.DataFrame({'Data': graph_HDBSCAN}).to_csv(os.path.join(folder_path, "graph_HDBSCAN.csv"), index=False)

        if graph_DBSCAN is not None:
            pd.DataFrame({'Data': graph_DBSCAN}).to_csv(os.path.join(folder_path, "graph_DBSCAN.csv"), index=False)

        if graph_Spect is not None:
            pd.DataFrame({'Data': graph_Spect}).to_csv(os.path.join(folder_path, "graph_SpectralClustering.csv"), index=False)

        if graph_Aggl is not None:
            pd.DataFrame({'Data': graph_Aggl}).to_csv(os.path.join(folder_path, "graph_AgglomerativeClustering.csv"), index=False)


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

            result = data_constructor(data, nrows, OPTICS_ON, OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample, DBSCAN_ON, 
                                DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample, HDBSCAN_ON, HDBSCAN_metric, 
                                HDBSCAN_algorithm, HDBSCAN_sample, Aggl_ON, Aggl_metric, Aggl_linkage, 
                                Aggl_n_clusters, Spect_ON, Spect_affinity, Spect_assign_labels, Spect_n_clusters)

            global gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
            TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN,TRACLUS_map_cluster_OPTICS, \
            TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_cluster_DBSCAN, \
            TRACLUS_map_cluster_Spect, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_HDBSCAN, \
            TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_Spect, \
            TRACLUS_map_df_Spect, TRACLUS_map_Aggl, TRACLUS_map_df_Aggl, \
            tabla_OPTICS, tabla_HDBSCAN, tabla_DBSCAN, tabla_Spect, tabla_Aggl, \
            graph_OPTICS, graph_HDBSCAN, graph_DBSCAN, graph_Spect, graph_Aggl, error_message

            gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
            TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN,TRACLUS_map_cluster_OPTICS, \
            TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_cluster_DBSCAN, \
            TRACLUS_map_cluster_Spect, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_HDBSCAN, \
            TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_Spect, \
            TRACLUS_map_df_Spect, TRACLUS_map_Aggl, TRACLUS_map_df_Aggl, \
            tabla_OPTICS, tabla_HDBSCAN, tabla_DBSCAN, tabla_Spect, tabla_Aggl, \
            graph_OPTICS, graph_HDBSCAN, graph_DBSCAN, graph_Spect, graph_Aggl, error_message, = result

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

    def process_csv_from_url(n_clicks_upload, url, nrows, folder_name):
        if n_clicks_upload is not None and n_clicks_upload > 0:
            if not folder_name:
                return dash.no_update, html.Div(["Por favor, introduce un nombre para el experimento."])
            if not url:
                return dash.no_update, html.Div(["No se ha introducido ningún enlace."])
            if not nrows:
                return dash.no_update, html.Div(["No se ha introducido el número de filas."])
            
            result = data_constructor(url, nrows, OPTICS_ON, OPTICS_metric, OPTICS_algorithm, OPTICS_eps, OPTICS_sample, DBSCAN_ON, 
                                DBSCAN_metric, DBSCAN_algorithm, DBSCAN_eps, DBSCAN_sample, HDBSCAN_ON, HDBSCAN_metric, 
                                HDBSCAN_algorithm, HDBSCAN_sample, Aggl_ON, Aggl_metric, Aggl_linkage, 
                                Aggl_n_clusters, Spect_ON, Spect_affinity, Spect_assign_labels, Spect_n_clusters)

            global gdf, tray, html_map, html_heatmap
            global TRACLUS_map_OPTICS, TRACLUS_map_cluster_OPTICS, TRACLUS_map_df_OPTICS, tabla_OPTICS, graph_OPTICS
            global TRACLUS_map_HDBSCAN, TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_df_HDBSCAN, tabla_HDBSCAN, graph_HDBSCAN
            global TRACLUS_map_DBSCAN, TRACLUS_map_cluster_DBSCAN, TRACLUS_map_df_DBSCAN, tabla_DBSCAN, graph_DBSCAN
            global TRACLUS_map_Spect, TRACLUS_map_cluster_Spect, TRACLUS_map_df_Spect, tabla_Spect, graph_Spect
            global TRACLUS_map_Aggl, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_Aggl, tabla_Aggl, graph_Aggl
            global error_message

            gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
            TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN,TRACLUS_map_cluster_OPTICS, \
            TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_cluster_DBSCAN, \
            TRACLUS_map_cluster_Spect, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_HDBSCAN, \
            TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_Spect, \
            TRACLUS_map_df_Spect, TRACLUS_map_Aggl, TRACLUS_map_df_Aggl, \
            tabla_OPTICS, tabla_HDBSCAN, tabla_DBSCAN, tabla_Spect, tabla_Aggl, \
            graph_OPTICS, graph_HDBSCAN, graph_DBSCAN, graph_Spect, graph_Aggl, error_message = result

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
            if OPTICS_ON:
                return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_cluster_OPTICS,TRACLUS_map_df_OPTICS)
            elif HDBSCAN_ON:
                return get_clusters_map(TRACLUS_map_HDBSCAN, TRACLUS_map_cluster_HDBSCAN,TRACLUS_map_df_HDBSCAN)
            elif DBSCAN_ON:
                return get_clusters_map(TRACLUS_map_DBSCAN, TRACLUS_map_cluster_DBSCAN, TRACLUS_map_df_DBSCAN)
            elif Spect_ON:
                return get_clusters_map(TRACLUS_map_Spect, TRACLUS_map_cluster_Spect, TRACLUS_map_df_Spect)
            elif Aggl_ON:
                return get_clusters_map(TRACLUS_map_Aggl, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_Aggl)
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'item-1-1':
                return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_cluster_OPTICS, TRACLUS_map_df_OPTICS)
            elif button_id == 'item-1-2':
                return get_clusters_map(TRACLUS_map_HDBSCAN, TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_df_HDBSCAN)
            elif button_id == 'item-1-3':
                return get_clusters_map(TRACLUS_map_DBSCAN, TRACLUS_map_cluster_DBSCAN, TRACLUS_map_df_DBSCAN)
            elif button_id == 'item-1-4':
                return get_clusters_map(TRACLUS_map_Spect, TRACLUS_map_cluster_Spect,TRACLUS_map_df_Spect)
            elif button_id == 'item-1-5':
                return get_clusters_map(TRACLUS_map_Aggl, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_Aggl)
            
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
            if OPTICS_ON:
                return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_cluster_OPTICS,TRACLUS_map_df_OPTICS)
            elif HDBSCAN_ON:
                return get_clusters_map(TRACLUS_map_HDBSCAN, TRACLUS_map_cluster_HDBSCAN,TRACLUS_map_df_HDBSCAN)
            elif DBSCAN_ON:
                return get_clusters_map(TRACLUS_map_DBSCAN, TRACLUS_map_cluster_DBSCAN, TRACLUS_map_df_DBSCAN)
            elif Spect_ON:
                return get_clusters_map(TRACLUS_map_Spect, TRACLUS_map_cluster_Spect, TRACLUS_map_df_Spect)
            elif Aggl_ON:
                return get_clusters_map(TRACLUS_map_Aggl, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_Aggl)
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'item-1-1':
                return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_cluster_OPTICS, TRACLUS_map_df_OPTICS)
            elif button_id == 'item-1-2':
                return get_clusters_map(TRACLUS_map_HDBSCAN, TRACLUS_map_cluster_HDBSCAN, TRACLUS_map_df_HDBSCAN)
            elif button_id == 'item-1-3':
                return get_clusters_map(TRACLUS_map_DBSCAN, TRACLUS_map_cluster_DBSCAN, TRACLUS_map_df_DBSCAN)
            elif button_id == 'item-1-4':
                return get_clusters_map(TRACLUS_map_Spect, TRACLUS_map_cluster_Spect,TRACLUS_map_df_Spect)
            elif button_id == 'item-1-5':
                return get_clusters_map(TRACLUS_map_Aggl, TRACLUS_map_cluster_Aggl, TRACLUS_map_df_Aggl)

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
            if OPTICS_ON:
                return get_table(tabla_OPTICS)
            elif HDBSCAN_ON:
                return get_table(tabla_HDBSCAN)
            elif DBSCAN_ON:
                return get_table(tabla_DBSCAN)
            elif Spect_ON:
                return get_table(tabla_Spect)
            elif Aggl_ON:
                return get_table(tabla_Aggl)
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'table-1':
                return get_table(tabla_OPTICS)
            elif button_id == 'table-2':
                return get_table(tabla_HDBSCAN)
            elif button_id == 'table-3':
                return get_table(tabla_DBSCAN)
            elif button_id == 'table-4':
                return get_table(tabla_Spect)
            elif button_id == 'table-5':
                return get_table(tabla_Aggl)
            
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
                labels={'x': 'Clúster', 'y': 'Segmentos'}
            )

        # Filtrar los datos según la selección del usuario
        if selected_filter == 'optics' and OPTICS_ON:
            filtered_data = graph_OPTICS
        elif selected_filter == 'hdbscan' and HDBSCAN_ON:
            filtered_data = graph_HDBSCAN
        elif selected_filter == 'dbscan' and DBSCAN_ON:
            filtered_data = graph_DBSCAN
        elif selected_filter == 'spectral' and Spect_ON:
            filtered_data = graph_Spect
        elif selected_filter == 'agglomerative' and Aggl_ON:
            filtered_data = graph_Aggl
        else:
            # Si el filtro seleccionado no tiene datos o está deshabilitado
            return px.bar(
                title=f'No hay datos para {selected_filter.capitalize()}',
                labels={'x': 'Clúster', 'y': 'Segmentos'}
            )

        # Contar la frecuencia de cada clúster
        cluster_counts = Counter(filtered_data)
        
        # Crear el gráfico de barras con Plotly Express
        figure = px.bar(
            x=list(cluster_counts.keys()),
            y=list(cluster_counts.values()),
            labels={'x': 'Clúster', 'y': 'Segmentos'},
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
            if OPTICS_ON:
                txt_files["tabla_OPTICS.txt"] = tabla_OPTICS.to_csv(index=False, sep='\t')
                images["TRACLUS_map_OPTICS.png"] = TRACLUS_map_OPTICS
                images["TRACLUS_map_df_OPTICS.png"] = TRACLUS_map_df_OPTICS
            if HDBSCAN_ON:
                txt_files["tabla_HDBSCAN.txt"] = tabla_HDBSCAN.to_csv(index=False, sep='\t')
                images["TRACLUS_map_HDBSCAN.png"] = TRACLUS_map_HDBSCAN
                images["TRACLUS_map_df_HDBSCAN.png"] = TRACLUS_map_df_HDBSCAN
            if DBSCAN_ON:
                txt_files["tabla_DBSCAN.txt"] = tabla_DBSCAN.to_csv(index=False, sep='\t')
                images["TRACLUS_map_DBSCAN.png"] = TRACLUS_map_DBSCAN
                images["TRACLUS_map_df_DBSCAN.png"] = TRACLUS_map_df_DBSCAN
            if Spect_ON:
                txt_files["tabla_SpectralClustering.txt"] = tabla_Spect.to_csv(index=False, sep='\t')
                images["TRACLUS_map_SpectralClustering.png"] = TRACLUS_map_Spect
                images["TRACLUS_map_df_SpectralClustering.png"] = TRACLUS_map_df_Spect
            if Aggl_ON:
                txt_files["tabla_AgglomerativeClustering.txt"] = tabla_Aggl.to_csv(index=False, sep='\t')
                images["TRACLUS_map_AgglomerativeClustering.png"] = TRACLUS_map_Aggl
                images["TRACLUS_map_df_AgglomerativeClustering.png"] = TRACLUS_map_df_Aggl

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