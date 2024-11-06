import dash
import dash_bootstrap_components as dbc
from dash import *
import dash_table
from Funtions import map_ilustration, map_heat, solicitar_coordenadas
from Data_loading import constructor
import matplotlib
matplotlib.use('Agg')
import base64
import zipfile

# Pagina Carga de datos

def get_page_zero():
    return html.Div([
        html.Div([
            html.H1("Introducción de datos previos"),
        ], className='box title'),
        html.Div([
            html.Div([
                html.H2("Introduce el enlace del archivo que se va a analizar:"),
                dcc.Input(
                    id='input-url',
                    type='text',
                    placeholder='Introduce el enlace del archivo .csv',
                    className='file-upload'
                ),
            ], className='box inputfile'),
            html.Div([
                html.H2("Número de trayectorias que se van a usar:"),
                dcc.Input(
                    id='nrows-input',
                    type='number',
                    placeholder='Número de filas',
                    value='',
                    className='number-input'
                )
            ], className='box inputnumber'),
            html.Div([
                dbc.Button('Comenzar procesamiento', id='process-url-button', n_clicks=0)
            ], className='box buttonsconfirm'),
            html.Div([
                dbc.Button('Configuración predeterminada', id='default-config-button', n_clicks=0)
            ], className='box buttonsdefault'),
        ], className='grid-data-container'),
        html.Div([
            dbc.Spinner(children=[
                html.Div(id='output-container', className='box output')
            ]),
            dbc.Spinner(children=[
                html.Div(id='predeterminate-data', className='box output')
            ])
        ], className='box output'),
        html.Div([
            dcc.Store(id='data-store')
        ], className='box data-store')        
    ], className='gid-zero-container')

# Pagina mapa
def bytes_to_base64(image_bytes):
    image_bytes.seek(0)  # Asegúrate de que el puntero esté al principio
    return base64.b64encode(image_bytes.read()).decode('utf-8')

def get_map_image_as_html(html_map, html_heatmap):

    html_map = bytes_to_base64(html_map)
    html_heatmap = bytes_to_base64(html_heatmap)

    return html.Div([
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{html_map}",
                    className='image-rounded'
                ),
            ], className="container-map"),
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{html_heatmap}",
                    className='image-rounded'
                ),
            ], className="container-map")
        ], className="container-maps")

def get_home_page():
    items1 = [
        dbc.DropdownMenuItem("Item 1", id="option-1-1"), 
        dbc.DropdownMenuItem("Item 2", id="option-1-2"),
        dbc.DropdownMenuItem("Item 3", id="option-1-3")
    ]

    items2 = [
        dbc.DropdownMenuItem("Item 1", id="option-2-1"), 
        dbc.DropdownMenuItem("Item 2", id="option-2-2"),
        dbc.DropdownMenuItem("Item 3", id="option-2-3")
    ]

    return html.Div([
        html.Div([
            dbc.DropdownMenu(
                items1, label="Cordenadas", color="primary"
            )
        ], className="box menu3"),
        html.Div([
            dbc.DropdownMenu(
                items2, label="Zoom", color="primary"
            )
        ], className="box menu4"),
        html.Div([
            dbc.Spinner(children=[html.Div(id='map-container')])
        ], className="box maps")
    ], className="grid-home-container")

# Pagina comparacion TRACLUS

# Ejemplo de uso para mostrar las imágenes en el carousel
def get_clusters_map(TRACLUS_map, TRACLUS_map_df):
    # Convertir las imágenes a base64 para mostrarlas en el navegador
    TRACLUS_map = bytes_to_base64(TRACLUS_map)
    TRACLUS_map_df = bytes_to_base64(TRACLUS_map_df)

    return html.Div([
        dbc.Carousel(
            items=[
                {"key": "1", "src": f"data:image/png;base64,{TRACLUS_map}"},
                {"key": "2", "src": f"data:image/png;base64,{TRACLUS_map_df}"}
            ],
            controls=True,
            indicators=True,
            variant="dark",
            className="image-carousel"
        )
    ], className="container-map")

def get_comparation_page():
    items1 = [
        dbc.DropdownMenuItem("OPTICS", id="item-1-1"), 
        dbc.DropdownMenuItem("HDBSCAN", id="item-1-2"),
        dbc.DropdownMenuItem("DBSCAN", id="item-1-3"),
        dbc.DropdownMenuItem("SpectralClustering", id="item-1-4"),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="item-1-5")
    ]

    items2 = [
        dbc.DropdownMenuItem("OPTICS", id="item-2-1"), 
        dbc.DropdownMenuItem("HDBSCAN", id="item-2-2"),
        dbc.DropdownMenuItem("DBSCAN", id="item-2-3"),
        dbc.DropdownMenuItem("SpectralClustering", id="item-2-4"),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="item-2-5")
    ]

    return html.Div([
        html.Div([
            dbc.DropdownMenu(
                items1, label="Algoritmo", color="primary"
            )
        ], className="box menu1"),
        html.Div([
            dbc.DropdownMenu(
                items2, label="Algoritmo", color="primary"
            )
        ], className="box menu2"),
        html.Div([
            dbc.Spinner(children=[html.Div(id='map-clusters-1')])  
        ], className="box map1"),  
        html.Div([
            dbc.Spinner(children=[html.Div(id='map-clusters-2')])  
        ], className="box map2")      
    ], className="grid-compratator-container")

# Pagina tablas

def get_table(tabla):
    # Convertir los valores que no son serializables a formato string
    if 'geometry' in tabla.columns:
        tabla['geometry'] = tabla['geometry'].apply(lambda geom: str(geom))

    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in tabla.columns],
        data=tabla.to_dict('records'),
        filter_action='native',
        sort_action='native',
        page_action='native',
        page_size=10,
        style_table={'overflowX': 'auto'},
    )

def get_estadistic_page():
    item_table = [
        dbc.DropdownMenuItem("OPTICS", id="table-1"), 
        dbc.DropdownMenuItem("HDBSCAN", id="table-2"),
        dbc.DropdownMenuItem("DBSCAN", id="table-3"),
        dbc.DropdownMenuItem("SpectralClustering", id="table-4"),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="table-5")
    ]

    return html.Div([
        dbc.Container([
            html.H1("Tabla Interactiva en Dash", className="text-center my-3"),
            html.Div([
                dbc.DropdownMenu(
                    item_table, label="Algoritmo de la tabla", color="primary"
                )
            ], className="box menu1"),
            dcc.Store(id='stored-data'),  # Almacenamiento en el lado del cliente
            html.Div([
                dbc.Spinner(children=[html.Div(id='table-container')])  
            ], className="box map1")          
        ])
    ])

# Creación de la aplicación Dash

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

# app.server.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5 GB en bytes

# Definición del layout principal de la aplicación utilizando componentes de Dash
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Navbar(
        dbc.Container(children=[
            dbc.NavItem(dbc.NavLink("TRACLUS", href="/", className="navbar-text-title")),
            dbc.NavItem(dbc.NavLink("Mapa de trayectorias", href="/home", className="navbar-text")),
            dbc.NavItem(dbc.NavLink("Comparacion de algoritmos", href="/comparacion", className="navbar-text")),
            dbc.NavItem(dbc.NavLink("Estadísticas", href="/estadisticas", className="navbar-text")),  
            # Botón de descarga de datos
            dbc.NavItem([
                dbc.Button("Descargar Datos", id="btn-download-txt", className="navbar-text"),
                dcc.Download(id="download-text")
            ])
        ]),
        color="success",
        className="header-navbar"
    ),
    html.Div(id='page-content', className="page-content")  
], className="grid-main-container")

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/home':
        return get_home_page() 
    elif pathname == '/comparacion':
        return get_comparation_page()
    elif pathname == '/estadisticas':
        return get_estadistic_page() 
    else:
        return get_page_zero()
    
@app.callback(
    Output('map-clusters-1', 'children'),
    [Input('item-1-1', 'n_clicks'),
    Input('item-1-2', 'n_clicks'),
    Input('item-1-3', 'n_clicks'),
    Input('item-1-4', 'n_clicks'),
    Input('item-1-5', 'n_clicks')]
)
def display_clusters_1(*args):
    ctx = dash.callback_context

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
    ctx = dash.callback_context

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

@app.callback(
    Output('map-container', 'children'),
    [Input('option-1-1', 'n_clicks'),
    Input('option-1-2', 'n_clicks'),
    Input('option-1-3', 'n_clicks')]
)

def update_map(*args):
    ctx = dash.callback_context

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

@app.callback(
    Output('predeterminate-data', 'children'),
    [Input('default-config-button', 'n_clicks')],
    prevent_initial_call=True
)

def update_output_predeter(n_clicks):
    if n_clicks > 0:
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

        return (html.Div(['Archivo cargado y procesado con configuración predeterminada.']),
            html.Div([f'{nrows} filas cargadas desde {data}']))

@app.callback(
    [Output('output-container', 'children'),
    Output('data-store', 'data')],
    Input('process-url-button', 'n_clicks'),
    State('input-url', 'value'),
    State('nrows-input', 'value'),
    prevent_initial_call=True
)

def process_csv_from_url(n_clicks, url, nrows):
    if n_clicks > 0:
        
        if not url:
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

        return (html.Div(['Archivo cargado y procesado con configuración personalizada.']),
            html.Div([f'{nrows} filas cargadas desde {url}']))

""" @app.callback(
    [Output('output-container', 'children')],
    [Input('confirm-button', 'n_clicks'), 
    Input('default-config-button', 'n_clicks1')],
    [State('nrows-input', 'value'),
    State('upload-data', 'contents')],
    prevent_initial_call=True
)

def update_output(n_clicks, n_clicks1, nrows, contents):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "No se ha seleccionado ninguna configuración.", "No se ha seleccionado ninguna configuración."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'confirm-button':
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in content_type:
                    # Asume que el usuario sube un CSV
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    return html.Div([
                        html.P(f"Número de trayectorias: {nrows}"),
                        html.P(f"Primeras filas del archivo CSV:"),
                        dcc.Graph(
                            figure={
                                'data': [{'x': df.index, 'y': df[col], 'type': 'line', 'name': col} for col in df.columns],
                                'layout': {'title': 'Datos del CSV'}
                            }
                        )
                    ])
                else:
                    return 'El archivo no es un CSV.'
            except Exception as e:
                print(e)
                return 'Hubo un error al procesar el archivo.'
        return 'No se ha subido ningún archivo.'
    elif button_id == 'default-config-button':
        nrows = 50000
        filename = "C:/Users/Álvaro/Documents/GitHub/TFG/TFG_TRACLUS/app/train_data/taxis_trajectory/train.csv"
        gdf = load_and_simplify_data(filename, nrows)
        return (html.Div(['Archivo cargado y procesado con configuración predeterminada.']),
                html.Div([f'{nrows} filas cargadas desde {filename}'])) """

""" def update_output(n_clicks, contents, nrows, filename):


    if n_clicks > 0:
        
        try:
            global gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_SpectralClustering, TRACLUS_map_df_SpectralClustering, TRACLUS_map_AgglomerativeClustering, TRACLUS_map_df_AgglomerativeClustering

            gdf, tray, html_map, html_heatmap, TRACLUS_map_OPTICS, TRACLUS_map_df_OPTICS, TRACLUS_map_HDBSCAN, TRACLUS_map_df_HDBSCAN, TRACLUS_map_DBSCAN, TRACLUS_map_df_DBSCAN, TRACLUS_map_SpectralClustering, TRACLUS_map_df_SpectralClustering, TRACLUS_map_AgglomerativeClustering, TRACLUS_map_df_AgglomerativeClustering = constructor (filename, nrows)
                    
            return (html.Div(['Archivo cargado y procesado con configuración predeterminada.']),
                html.Div([f'{nrows} filas cargadas desde NOMBREDELARCHIVO']))
        except Exception as e:
            print(e)
            return 'No se ha introducido los datos correctamente.'



        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in content_type:
                    # Asume que el usuario sube un CSV
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    return html.Div([
                        html.P(f"Número de trayectorias: {nrows}"),
                        html.P(f"Primeras filas del archivo CSV:"),
                        dcc.Graph(
                            figure={
                                'data': [{'x': df.index, 'y': df[col], 'type': 'line', 'name': col} for col in df.columns],
                                'layout': {'title': 'Datos del CSV'}
                            }
                        )
                    ])
                else:
                    return 'El archivo no es un CSV.'
            except Exception as e:
                print(e)
                return 'Hubo un error al procesar el archivo.'
        return 'No se ha subido ningún archivo.'
    #return 'No hay datos para mostrar.' """

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
    ctx = dash.callback_context

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
        
#* Descargar datos en formato ZIP

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

        ## Agregar imágenes PNG al ZIP
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