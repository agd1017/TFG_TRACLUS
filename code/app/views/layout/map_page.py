import dash_bootstrap_components as dbc
from dash import html

from utils.data_utils import bytes_to_base64

def get_map_image_as_html(html_map, html_heatmap):
    """
    Converts map and heatmap images into base64-encoded HTML format and returns 
    the images wrapped in HTML elements for display.

    Args:
        html_map (bytes): The image data for the map to be displayed.
        html_heatmap (bytes): The image data for the heatmap to be displayed.

    Returns:
        html.Div: A Dash HTML layout containing two images (map and heatmap) wrapped in divs.
    """
    
    # Convert images to base64 encoding to embed in the HTML
    html_map = bytes_to_base64(html_map)
    html_heatmap = bytes_to_base64(html_heatmap)

    # Return the HTML layout with the images of the map and heatmap
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

def get_page_map():
    """
    Creates the layout for the map page, which includes a spinner to show loading 
    content and a container for the map visualization.

    Returns:
        html.Div: A Dash layout for the map page containing a spinner and map container.
    """
    
    # Return the layout for the map page with a spinner for loading
    return html.Div([
        html.Div([
            dbc.Spinner(children=[html.Div(id='map-container')])
        ], className="box maps")
    ], className="grid-map-container")