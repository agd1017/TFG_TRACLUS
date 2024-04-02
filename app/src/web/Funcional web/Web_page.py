import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc, State
import base64
import io
import pandas as pd
import dash_table
from Funtions import map_ilustration, map_heat, solicitar_coordenadas, load_and_simplify_data, create_dataframe
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
        html.Div([
            html.H1("Introducción de datos previos"),
        ], className='box title'),
        html.Div([
            html.Div([
                html.H2("Archivo que se va a analizar:"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Arrastra o selecciona un archivo']),
                    className='file-upload',
                    accept='.csv'
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
                dbc.Button('Configuración selecionada', id='confirm-button', n_clicks=0)
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
        ], className='box output')
    ], className='gid-zero-container')

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

def get_estadistic_page():

    return html.Div([
        dbc.Container([
            html.H1("Tabla Interactiva en Dash", className="text-center my-3"),
            dbc.Button("Actualizar Datos", id="refresh-button", className="mb-3"),
            dcc.Store(id='stored-data'),  # Almacenamiento en el lado del cliente
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in create_dataframe().columns],
                        filter_action='native',
                        sort_action='native',
                        page_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                    ), width=12
                )
            )
        ], fluid=True)  
    ])
        
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Creación de la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

# Definición del layout principal de la aplicación utilizando componentes de Dash
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Navbar(
        dbc.Container(children=[
            dbc.NavItem(dbc.NavLink("TRACLUS", href="/", className="navbar-text-title")),
            dbc.NavItem(dbc.NavLink("Mapa de trayectorias", href="/home", className="navbar-text")),
            dbc.NavItem(dbc.NavLink("Comparacion de algoritmos", href="/comparacion", className="navbar-text")),
            dbc.NavItem(dbc.NavLink("Estadísticas", href="/estadisticas", className="navbar-text"))  
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

@app.callback(
    Output('predeterminate-data', 'children'),
    [Input('default-config-button', 'n_clicks')],
    prevent_initial_call=True
)

def update_output_predeter(n_clicks):
    if n_clicks > 0:
        nrows = 50000
        filename = "C:/Users/Álvaro/Documents/GitHub/TFG/TFG_TRACLUS/app/train_data/taxis_trajectory/train.csv"
        gdf = load_and_simplify_data(filename, nrows)
        return (html.Div(['Archivo cargado y procesado con configuración predeterminada.']),
                html.Div([f'{nrows} filas cargadas desde {filename}']))

# Callback para capturar la entrada y mostrarla
@app.callback(
    Output('output-container', 'children'),
    [Input('confirm-button', 'n_clicks')],
    [State('nrows-input', 'value'),
    State('upload-data', 'contents')],
    prevent_initial_call=True
)

def update_output(n_clicks, nrows, contents):
    if n_clicks > 0:
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
    return 'No hay datos para mostrar.'

@app.callback(
    Output('stored-data', 'data'),
    Input('refresh-button', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def update_stored_data(n_clicks, data):
    if n_clicks is None or n_clicks == 0:
        # Esto evita que la función se ejecute en la carga inicial
        raise dash.exceptions.PreventUpdate
    df = create_dataframe()  # Obtener los datos actualizados
    return df.to_dict('records')

@app.callback(
    Output('table', 'data'),
    Input('stored-data', 'data'),
    prevent_initial_call=True  # Esto evita que la tabla se llene dos veces en la carga inicial
)
def update_table(data):
    if data is None:
        # Carga inicial de datos
        return create_dataframe().to_dict('records')
    return data

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
    # http://127.0.0.1:8050/