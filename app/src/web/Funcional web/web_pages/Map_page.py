import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')
import base64

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

def get_page_map():
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