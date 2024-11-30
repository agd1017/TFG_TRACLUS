import dash_bootstrap_components as dbc
from dash import html
import matplotlib
matplotlib.use('Agg')

from utils.data_utils import bytes_to_base64

# Pagina mapa

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

def get_page_map():
    
    return html.Div([
        html.Div([
            dbc.Spinner(children=[html.Div(id='map-container')])
        ], className="box maps")
    ], className="grid-map-container")