import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')
import base64

# Pagina comparacion TRACLUS
def bytes_to_base64(image_bytes):
    image_bytes.seek(0)  # Asegúrate de que el puntero esté al principio
    return base64.b64encode(image_bytes.read()).decode('utf-8')

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

def get_page_mapTRACLUS():
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