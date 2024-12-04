import dash_bootstrap_components as dbc
from dash import html

from utils.data_utils import bytes_to_base64
    
def get_clusters_map(traclus_map, traclus_map_cluster, traclus_map_segments):
    """
    Converts the given maps to base64 and returns an HTML layout for displaying them 
    in a carousel.

    The images are displayed using Dash's Carousel component, allowing users to 
    cycle through different visualizations (original map, map with clusters, and map with segments).

    Args:
        traclus_map (bytes): The raw image data for the original TRACLUS map.
        traclus_map_cluster (bytes): The raw image data for the map with clusters.
        traclus_map_segments (bytes): The raw image data for the map with segments.

    Returns:
        html.Div: The layout with the carousel displaying the images.
    """
    # Convert the images to base64 for embedding in the browser
    traclus_map = bytes_to_base64(traclus_map)
    traclus_map_cluster = bytes_to_base64(traclus_map_cluster)
    traclus_map_segments = bytes_to_base64(traclus_map_segments)
    
    return html.Div([
        dbc.Carousel(  # Carousel to display the images
            items=[
                {"key": "1", "src": f"data:image/png;base64,{traclus_map}"},
                {"key": "2", "src": f"data:image/png;base64,{traclus_map_cluster}"},
                {"key": "3", "src": f"data:image/png;base64,{traclus_map_segments}"}
            ],
            controls=True,  # Enable navigation controls (previous/next buttons)
            indicators=True,  # Enable indicators (dots at the bottom)
            variant="dark",  # Dark variant for the carousel
            className="image-carousel"  # Custom CSS class for styling
        )
    ], className="container-map")  # Custom container for the carousel

def get_page_maptraclus(optics_on, hdbscan_on, dbscan_on, spect_on, aggl_on):
    """
    Creates the layout for the TRACLUS map page, including dropdown menus for selecting 
    clustering algorithms and displaying the corresponding map images.

    Args:
        optics_on (bool): Whether the OPTICS option is enabled.
        hdbscan_on (bool): Whether the HDBSCAN option is enabled.
        dbscan_on (bool): Whether the DBSCAN option is enabled.
        spect_on (bool): Whether the Spectral Clustering option is enabled.
        aggl_on (bool): Whether the Agglomerative Clustering option is enabled.

    Returns:
        html.Div: The layout for the TRACLUS map page with dropdown menus and map displays.
    """
    # Dropdown items for the first algorithm selection
    items1 = [
        dbc.DropdownMenuItem("OPTICS", id="item-1-1", disabled=not optics_on), 
        dbc.DropdownMenuItem("HDBSCAN", id="item-1-2", disabled=not hdbscan_on),
        dbc.DropdownMenuItem("DBSCAN", id="item-1-3", disabled=not dbscan_on),
        dbc.DropdownMenuItem("SpectralClustering", id="item-1-4", disabled=not spect_on),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="item-1-5", disabled=not aggl_on)
    ]

    # Dropdown items for the second algorithm selection
    items2 = [
        dbc.DropdownMenuItem("OPTICS", id="item-2-1", disabled=not optics_on), 
        dbc.DropdownMenuItem("HDBSCAN", id="item-2-2", disabled=not hdbscan_on),
        dbc.DropdownMenuItem("DBSCAN", id="item-2-3", disabled=not dbscan_on),
        dbc.DropdownMenuItem("SpectralClustering", id="item-2-4", disabled=not spect_on),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="item-2-5", disabled=not aggl_on)
    ]

    return html.Div([  # Main container for the layout
        html.Div([  # First dropdown menu for selecting the algorithm
            dbc.DropdownMenu(
                items1, label="Algoritmo", color="primary"
            )
        ], className="box menu1"),
        html.Div([  # Second dropdown menu for selecting the algorithm
            dbc.DropdownMenu(
                items2, label="Algoritmo", color="primary"
            )
        ], className="box menu2"),
        html.Div([  # First map display container with a spinner while loading
            dbc.Spinner(children=[html.Div(id='map-clusters-1')])  
        ], className="box map1"),  
        html.Div([  # Second map display container with a spinner while loading
            dbc.Spinner(children=[html.Div(id='map-clusters-2')])  
        ], className="box map2")      
    ], className="grid-compratator-container")  # Custom grid container for layout