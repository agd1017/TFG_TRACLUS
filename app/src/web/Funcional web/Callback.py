import dash
import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')
import zipfile
from Funtions import *
from web_pages.Experiment_page import *
from web_pages.Select_page import *
from web_pages.DataUpload_page import *
from web_pages.Map_page import *
from web_pages.TRACLUSMap_page import *
from web_pages.Table_page import *
from Data_loading import constructor

# Definición de la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

# Definición del layout principal de la aplicación utilizando componentes de Dash
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='navbar-container'),
    html.Div(id='page-content', className="page-content")  
], className="grid-main-container")

# Callbacks for nav bar

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
        return get_page_mapTRACLUS()
    elif pathname == '/estadisticas':
        return get_page_tables()
    else:
        return get_page_select()

# Callbacks for select page

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input('previous-exp-button', 'n_clicks'),
    Input('new-exp-button', 'n_clicks')],
    prevent_initial_call=True
)
def navigate_to_page(n_clicks_previous, n_clicks_new):
    if n_clicks_previous > 0:
        return '/previous-experiments'
    elif n_clicks_new > 0:
        return '/new-experiment'
    return '/'

# Callbacks for experiment page

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('execute-button', 'n_clicks'),
    prevent_initial_call=True
)
def navigate_to_page_dataupdate(n_clicks_data):
    if n_clicks_data is not None and n_clicks_data > 0:
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

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Output('output-container', 'data', allow_duplicate=True),
    Input('default-config-button', 'n_clicks'),
    prevent_initial_call=True
)

def upload_output_predeter(n_clicks_upload):
    if n_clicks_upload is not None and n_clicks_upload > 0:
        data = "C:/Users/Álvaro/Documents/GitHub/TFG/TFG_TRACLUS/app/train_data/taxis_trajectory/train.csv"
        nrows = 5

        result = constructor(data, nrows)

        global gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
        TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, \
        TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_SpectralClustering, \
        TRACLUS_map_df_SpectralClustering, TRACLUS_map_AgglomerativeClustering, \
        TRACLUS_map_df_AgglomerativeClustering, tabla_OPTICS, tabla_HDBSCAN,  \
        tabla_DBSCAN, tabla_SpectralClustering, tabla_AgglomerativeClustering

        gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
        TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, \
        TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_SpectralClustering, \
        TRACLUS_map_df_SpectralClustering, TRACLUS_map_AgglomerativeClustering, \
        TRACLUS_map_df_AgglomerativeClustering, tabla_OPTICS, tabla_HDBSCAN,  \
        tabla_DBSCAN, tabla_SpectralClustering, tabla_AgglomerativeClustering = result

        return '/map-page', {}

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Output('output-container', 'data', allow_duplicate=True),
    Input('process-url-button', 'n_clicks'),
    State('input-url', 'value'),
    State('nrows-input', 'value'),
    prevent_initial_call=True
)

def process_csv_from_url(n_clicks_upload, url, nrows):
    if n_clicks_upload is not None and n_clicks_upload > 0:
        
        """  if not url:
            return "No se ha introducido ningún enlace.", None
        if not nrows:
            return "No se ha introducido el número de filas.", None

        result = constructor(url, nrows)

        global gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
        TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, \
        TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_SpectralClustering, \
        TRACLUS_map_df_SpectralClustering, TRACLUS_map_AgglomerativeClustering, \
        TRACLUS_map_df_AgglomerativeClustering, tabla_OPTICS, tabla_HDBSCAN,  \
        tabla_DBSCAN, tabla_SpectralClustering, tabla_AgglomerativeClustering

        gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, \
        TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, \
        TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_SpectralClustering, \
        TRACLUS_map_df_SpectralClustering, TRACLUS_map_AgglomerativeClustering, \
        TRACLUS_map_df_AgglomerativeClustering, tabla_OPTICS, tabla_HDBSCAN,  \
        tabla_DBSCAN, tabla_SpectralClustering, tabla_AgglomerativeClustering = result   
        """
        return '/map-page', {}

# Callbacks for map page
    
@app.callback(
    Output('map-container', 'children'),
    [Input('option-1-1', 'n_clicks'),
    Input('option-1-2', 'n_clicks'),
    Input('option-1-3', 'n_clicks')]
)

def update_map(*args):
    ctx = callback_context

    if not ctx.triggered:
        map = html_map
        heatmap = html_heatmap # "Seleccione un elemento para el mapa 2."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'option-1-1':
            map = html_map
            heatmap = html_heatmap
        elif button_id == 'option-1-2':
            minx, miny, maxx, maxy = -8.689, 41.107, -8.560, 41.185
            # Define aquí cómo generas el mapa para la Opción 2
            map = map_ilustration(gdf, minx, miny, maxx, maxy)
            heatmap = map_heat(gdf, minx, miny, maxx, maxy)  
        elif button_id == 'option-1-3':
            minx, miny, maxx, maxy = solicitar_coordenadas(gdf)
            # Define aquí cómo generas el mapa para la Opción 3
            map = map_ilustration(gdf, minx, miny, maxx, maxy)
            heatmap = map_heat(gdf, minx, miny, maxx, maxy) 
        
    map_image = get_map_image_as_html(map, heatmap)  
        
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
        return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS) # "Seleccione un elemento para el mapa 1."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'item-1-1':
            return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS)
        elif button_id == 'item-1-2':
            return get_clusters_map(TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN)
        elif button_id == 'item-1-3':
            return get_clusters_map(TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN)
        elif button_id == 'item-1-4':
            return get_clusters_map(TRACLUS_map_SpectralClustering, TRACLUS_map_df_SpectralClustering)
        elif button_id == 'item-1-5':
            return get_clusters_map(TRACLUS_map_AgglomerativeClustering, TRACLUS_map_df_AgglomerativeClustering)
        
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
        return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS) # "Seleccione un elemento para el mapa 2."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'item-2-1':
            return get_clusters_map(TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS)
        elif button_id == 'item-2-2':
            return get_clusters_map(TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN)
        elif button_id == 'item-2-3':
            return get_clusters_map(TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN)
        elif button_id == 'item-2-4':
            return get_clusters_map(TRACLUS_map_SpectralClustering, TRACLUS_map_df_SpectralClustering,)
        elif button_id == 'item-2-5':
            return get_clusters_map(TRACLUS_map_AgglomerativeClustering, TRACLUS_map_df_AgglomerativeClustering)

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
        return get_table(tabla_OPTICS) # "Seleccione un elemento para el mapa 2."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'table-1':
            return get_table(tabla_OPTICS)
        elif button_id == 'table-2':
            return get_table(tabla_HDBSCAN)
        elif button_id == 'table-3':
            return get_table(tabla_DBSCAN)
        elif button_id == 'table-4':
            return get_table(tabla_SpectralClustering)
        elif button_id == 'table-5':
            return get_table(tabla_AgglomerativeClustering)

# Callbacks for download data

@app.callback(
    Output("download-text", "data"),
    Input("btn-download-txt", "n_clicks"),
    prevent_initial_call=True,
)

def func(n_clicks):
    zip_file_name = "table.zip"
    with zipfile.ZipFile(zip_file_name, mode="w") as zf:
        # Crear archivos TXT en memoria y agregarlos al ZIP
        txt_files = {
            "tabla_OPTICS.txt": tabla_OPTICS.to_csv(index=False, sep='\t'),
            "tabla_HDBSCAN.txt": tabla_HDBSCAN.to_csv(index=False, sep='\t'),
            "tabla_DBSCAN.txt": tabla_DBSCAN.to_csv(index=False, sep='\t'),
            "tabla_SpectralClustering.txt": tabla_SpectralClustering.to_csv(index=False, sep='\t'),
            "tabla_AgglomerativeClustering.txt": tabla_AgglomerativeClustering.to_csv(index=False, sep='\t'),
        }
        
        for filename, txt_content in txt_files.items():
            zf.writestr(filename, txt_content)

        # Agregar imágenes PNG al ZIP
        images = {
            "map.png": html_map,
            "heatmap.png": html_heatmap,
            "OPTICS.png": TRACLUS_map_OPTICS,
            "HDBSCAN.png": TRACLUS_map_HDBSCAN,
            "DBSCAN.png": TRACLUS_map_DBSCAN,
            "SpectralClustering.png": TRACLUS_map_SpectralClustering,
            "AgglomerativeClustering.png": TRACLUS_map_AgglomerativeClustering,
        }
        
        for img_name, img_buf in images.items():
            img_buf.seek(0)  # Asegúrate de que el puntero esté al inicio del buffer
            zf.writestr(img_name, img_buf.getvalue())  # Agregar la imagen al ZIP
        
    return dcc.send_file(zip_file_name)

























if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
    # http://127.0.0.1:8050/