import dash_bootstrap_components as dbc
from dash import *
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

def get_page_tables(OPTICS_ON, HDBSCAN_ON, DBSCAN_ON, Spect_ON, Aggl_ON):
    item_table = [
        dbc.DropdownMenuItem("OPTICS", id="table-1", disabled=not OPTICS_ON), 
        dbc.DropdownMenuItem("HDBSCAN", id="table-2", disabled=not HDBSCAN_ON),
        dbc.DropdownMenuItem("DBSCAN", id="table-3", disabled=not DBSCAN_ON),
        dbc.DropdownMenuItem("SpectralClustering", id="table-4", disabled=not Spect_ON),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="table-5", disabled=not Aggl_ON)
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