import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc
from Funtions import map_ilustration, map_heat, solicitar_coordenadas, load_and_simplify_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def provisional():
    global gdf
    gdf = load_and_simplify_data("C:/Users/Álvaro/Documents/GitHub/TFG/TFG_TRACLUS/app/train_data/taxis_trajectory/train.csv", 100000)
    html_map = map_ilustration(gdf, -8.689, 41.107, -8.560, 41.185)
    html_heatmap = map_heat(gdf, -8.689, 41.107, -8.560, 41.185)

    return html_map, html_heatmap

html_map, html_heatmap = provisional()

def get_page_zero():
    return html.Div([
        dbc.Container([
            html.H1("Introducción de datos previos", className="text-center mb-4"),  # Añadido margen inferior (mb-4)
            dbc.Row(
                dbc.Col([
                    html.H2("Archivo que se va a analizar:", className="text-center"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Arrastra o selecciona un archivo'], className="text-center"),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'margin': '10px auto', 'display': 'block'
                        },
                        className='mb-3',  # Añadido margen inferior
                    ),
                ], width=12)
            ),
            dbc.Row(
                dbc.Col([
                        html.H2("Número de trayectorias que se van a usar:", className="text-center"),
                        dcc.Input(
                            id='nrows-input', 
                            type='number', 
                            placeholder='Número de filas', 
                            value='',
                            className='form-control',  # Estilo de Bootstrap para inputs
                            style={'margin': '0 auto', 'width': '50%'},  # Centrar y ajustar ancho del input
                        )
                ], width=12, className="mb-3")  # Añadido margen inferior        
            ),
            dbc.Row(
                dbc.Col([
                        dbc.Button('Confirmar', id='confirm-button', n_clicks=0, className='me-2'),  # Botón con margen
                        dbc.Button('Configuración predeterminada', id='default-config-button', n_clicks=0),
                        html.Div(id='output-container', className='mt-3')  # Margen superior para el contenedor
                ], width=12, className='text-center')
            ),
            dbc.Spinner(html.Div(id='loading-output-container'))
        ], fluid=True)      
    ])

# Pagina home
def get_map_image_as_html(html_map, html_heatmap):
    return html.Div([
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{html_map}",
                    style={
                        'maxHeight': '80vh',  # Altura máxima para evitar que la imagen sea demasiado alta
                        'width': '100%',      # Ancho ajustado al 100% del contenedor
                        'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto'
                    }
                ),
            ], style={'flex': '1', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'paddingRight': '5px'}),
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{html_heatmap}",
                    style={
                        'maxHeight': '80vh',  # Altura máxima para evitar que la imagen sea demasiado alta
                        'width': '100%',      # Ancho ajustado al 100% del contenedor
                        'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto'
                    }
                ),
            ], style={'flex': '1', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'paddingLeft': '5px'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'})

def get_home_page():
    return html.Div([
        html.Div([
            html.H4("Ajustes", className="mt-3"),
            html.Hr(),
            dbc.Label("Choose latitude and longitude"),
            dbc.RadioItems(
                options=[
                    {"label": "Option 1", "value": 1},
                    {"label": "Option 2", "value": 2},
                    {"label": "Option 3", "value": 3},
                ],
                value=1,
                id="lat-long-radio-items",
            ),
            html.Hr(),
            dbc.Label("Choose zoom and position of the map"),
            dbc.RadioItems(
                options=[
                    {"label": "Option 1", "value": 1},
                    {"label": "Option 2", "value": 2},
                    {"label": "Option 3", "value": 3},
                ],
                value=1,
                id="zoom-position-radio-items",
            ),
        ], className="box menusel"),
        html.Div([
            dcc.Loading(children=[html.Div(id='map-container')], type="circle")  
        ], className="box maps")
    ], className="grid-home-container")

# Pagina comparacion
def get_clusters_map():
    """ minx, miny, maxx, maxy = solicitar_coordenadas(gdf)
    html_heatmap = map_heat(gdf, minx, miny, maxx, maxy) """

    return html.Div([
        dbc.Carousel(
            items=[
                {"key": "1", "src": f"data:image/png;base64,{html_heatmap}"},
                {"key": "2", "src": f"data:image/png;base64,{html_map}"}
            ],
            controls=True,
            indicators=True,
            variant="dark"
        )
    ], className="container")

def get_comparation_page():
    items1 = [
        dbc.DropdownMenuItem("Item 1", id="item-1-1"), 
        dbc.DropdownMenuItem("Item 2", id="item-1-2"),
        dbc.DropdownMenuItem("Item 3", id="item-1-3")
    ]

    items2 = [
        dbc.DropdownMenuItem("Item 1", id="item-2-1"), 
        dbc.DropdownMenuItem("Item 2", id="item-2-2"),
        dbc.DropdownMenuItem("Item 3", id="item-2-3")
    ]

    return html.Div([
        html.Div([
            html.H1("Comparación de clusters")
        ], className="box title"),
        html.Div([
            dbc.DropdownMenu(
                items1, label="Algoritmo", color="primary",
            )
        ], className="box menu1"),
        html.Div([
            dbc.DropdownMenu(
                items2, label="Algoritmo", color="primary",
            )
        ], className="box menu2"),
        html.Div([
            dcc.Loading(children=[html.Div(id="map-clusters-1")], type="circle")  
        ], className="box map1"),  
        html.Div([
            dcc.Loading(children=[html.Div(id="map-clusters-2")], type="circle")  
        ], className="box map2")      
    ], className="grid-compratator-container")

# Creación de la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.LUMEN])

# Definición del layout principal de la aplicación utilizando componentes de Dash
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Navbar(
        dbc.Container(children=[
            html.Div([
                dbc.NavbarBrand("TRACLUS"),
            ], className="navbar-text"),
            html.Div([
                dbc.DropdownMenu(
                    label="Menú",
                    children=[
                        dbc.DropdownMenuItem("Carga de datos", href="/"),
                        dbc.DropdownMenuItem("Inicio", href="/home"),
                        dbc.DropdownMenuItem("Comparación", href="/comparacion"),
                        dbc.DropdownMenuItem("Estadísticas", href="/estadisticas"),
                    ]
                ),
            ]),   
        ]),
        color="primary",
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
        return html.Div("Página de Estadísticas")  
    else:
        return get_page_zero()
    
@app.callback(
    Output('map-clusters-1', 'children'),
    [Input('item-1-1', 'n_clicks'),
    Input('item-1-2', 'n_clicks'),
    Input('item-1-3', 'n_clicks')]
)
def display_clusters_1(*args):
    ctx = dash.callback_context

    if not ctx.triggered:
        return get_clusters_map() # "Seleccione un elemento para el mapa 1."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'item-1-1':
            return get_clusters_map()
        elif button_id == 'item-1-2':
            return get_clusters_map()
        elif button_id == 'item-1-3':
            return get_clusters_map()
        
@app.callback(
    Output('map-clusters-2', 'children'),
    [Input('item-2-1', 'n_clicks'),
    Input('item-2-2', 'n_clicks'),
    Input('item-2-3', 'n_clicks')]
)

def display_clusters_2(*args):
    ctx = dash.callback_context

    if not ctx.triggered:
        return get_clusters_map() # "Seleccione un elemento para el mapa 2."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'item-2-1':
            return get_clusters_map()
        elif button_id == 'item-2-2':
            return get_clusters_map()
        elif button_id == 'item-2-3':
            return get_clusters_map()
    
@app.callback(
    Output('map-container', 'children'),
    [Input('lat-long-radio-items', 'value')]  # Escucha los cambios de latitud y longitud
    #Input('zoom-position-radio-items', 'value')]  # Escucha los cambios de zoom y posición
)

def update_map(lat_long_value):
    # Aquí deberías decidir qué mapa mostrar basado en los valores seleccionados
    # Este es un pseudocódigo, necesitarás ajustar la lógica según tus datos y necesidades específicas
    if lat_long_value == 3:
        minx, miny, maxx, maxy = solicitar_coordenadas(gdf)
        # Define aquí cómo generas el mapa para la Opción 3
        map = map_ilustration(gdf, minx, miny, maxx, maxy)
        heatmap = map_heat(gdf, minx, miny, maxx, maxy)  
        
    elif lat_long_value == 2:
        minx, miny, maxx, maxy = -8.689, 41.107, -8.560, 41.185
        # Define aquí cómo generas el mapa para la Opción 2
        map = map_ilustration(gdf, minx, miny, maxx, maxy)
        heatmap = map_heat(gdf, minx, miny, maxx, maxy)  

    else:
        # Define aquí cómo generas el mapa para la Opción 1
        map = html_map
        heatmap = html_heatmap

    map_image = get_map_image_as_html(map, heatmap)

    return [map_image]

@app.callback(
    [Output('upload-data', 'children'),
    Output('output-container', 'children')],
    [Input('default-config-button', 'n_clicks')],
    prevent_initial_call=True  # Evita que la callback se ejecute al cargar la página
)
def update_output(n_clicks):
    #global gdf

    if n_clicks is None:
        # Esto evita que el callback se ejecute en el inicio antes de que cualquier botón sea presionado.
        return dash.no_update

    # Número de filas a leer del archivo CSV
    nrows = 100000
    # Ruta del archivo CSV
    filename = "C:/Users/Álvaro/Documents/GitHub/TFG/TFG_TRACLUS/app/train_data/taxis_trajectory/train.csv"


    # Asumiendo que 'load_and_simplify_data' es una función definida que carga y procesa tus datos
    gdf = load_and_simplify_data(filename, nrows)

    # Actualiza el contenido del componente 'upload-data' y 'output-container'
    return (html.Div(['Archivo cargado y procesado con configuración predeterminada.']),
            html.Div([f'{nrows} filas cargadas desde {filename}']))

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
    # http://127.0.0.1:8050/