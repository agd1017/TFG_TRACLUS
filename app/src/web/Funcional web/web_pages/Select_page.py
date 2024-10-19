import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')

# Pagina Experimento

def get_page_select():
    return html.Div([
        # Título
        html.Div([
            html.H1("Seleccione una de las opciones"),
        ], className='title'),

        # Contenedor de botones, con flexbox para centrarlos horizontalmente
        html.Div([
            html.Div([
                dbc.Button('Acceder a experimentos anteriores', id='previous-exp-button', n_clicks=0, className='button')
            ], className='button1-container'),

            html.Div([
                dbc.Button('Crear nuevo experimento', id='new-exp-button', n_clicks=0, className='button')
            ], className='button2-container')
        ], className='buttons-wrapper')
    ], className='grid-select-container')