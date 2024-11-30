import dash_bootstrap_components as dbc
from dash import html

from utils.data_utils import bytes_to_base64
    
# Ejemplo de uso para mostrar las imágenes en el carousel
def get_clusters_map(traclus_map, traclus_map_cluster, traclus_map_segments):
    # Convertir las imágenes a base64 para mostrarlas en el navegador
    traclus_map = bytes_to_base64(traclus_map)
    traclus_map_cluster = bytes_to_base64(traclus_map_cluster)
    traclus_map_segments = bytes_to_base64(traclus_map_segments)
    
    return html.Div([
        dbc.Carousel(
            items=[
                {"key": "1", "src": f"data:image/png;base64,{traclus_map}"},
                {"key": "2", "src": f"data:image/png;base64,{traclus_map_cluster}"},
                {"key": "3", "src": f"data:image/png;base64,{traclus_map_segments}"}
            ],
            controls=True,
            indicators=True,
            variant="dark",
            className="image-carousel"
        )
    ], className="container-map")

def get_page_maptraclus(optics_on, hdbscan_on, dbscan_on, spect_on, aggl_on):
    items1 = [
        dbc.DropdownMenuItem("OPTICS", id="item-1-1", disabled=not optics_on), 
        dbc.DropdownMenuItem("HDBSCAN", id="item-1-2", disabled=not hdbscan_on),
        dbc.DropdownMenuItem("DBSCAN", id="item-1-3", disabled=not dbscan_on),
        dbc.DropdownMenuItem("SpectralClustering", id="item-1-4", disabled=not spect_on),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="item-1-5", disabled=not aggl_on)
    ]

    items2 = [
        dbc.DropdownMenuItem("OPTICS", id="item-2-1", disabled=not optics_on), 
        dbc.DropdownMenuItem("HDBSCAN", id="item-2-2", disabled=not hdbscan_on),
        dbc.DropdownMenuItem("DBSCAN", id="item-2-3", disabled=not dbscan_on),
        dbc.DropdownMenuItem("SpectralClustering", id="item-2-4", disabled=not spect_on),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="item-2-5",disabled=not aggl_on)
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