import dash_bootstrap_components as dbc
from dash import dash_table, html, dcc
import matplotlib
matplotlib.use('Agg')

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

def get_page_tables(optics_on, hdbscan_on, dbscan_on, spect_ON, aggl_ON):
    item_table = [
        dbc.DropdownMenuItem("OPTICS", id="table-1", disabled=not optics_on), 
        dbc.DropdownMenuItem("HDBSCAN", id="table-2", disabled=not hdbscan_on),
        dbc.DropdownMenuItem("DBSCAN", id="table-3", disabled=dbscan_on),
        dbc.DropdownMenuItem("SpectralClustering", id="table-4", disabled=not spect_ON),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="table-5", disabled=not aggl_ON)
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
            ], className="box map1") ,
            html.Div([
                html.Label("Selecciona Clústeres a Mostrar:"),
                dcc.Dropdown(
                    id='cluster-selector',
                    options=[
                        {'label': 'OPTICS', 'value': 'optics', 'disabled': not optics_on},
                        {'label': 'HDBSCAN', 'value': 'hdbscan', 'disabled': not hdbscan_on},
                        {'label': 'DBSCAN', 'value': 'dbscan', 'disabled': dbscan_on},
                        {'label': 'Spectral Clustering', 'value': 'spectral', 'disabled': not spect_ON},
                        {'label': 'Agglomerative Clustering', 'value': 'agglomerative', 'disabled': not aggl_ON}
                    ],
                    value=None
                ),
                dcc.Graph(id='cluster-bar-chart')
            ])
        ])
    ])