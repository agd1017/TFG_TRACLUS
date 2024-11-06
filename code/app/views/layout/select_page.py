import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')
from utils.data_utils import list_experiment_folders

# Pagina Experimento

def get_page_select():
    return html.Div([
        # Título
        html.Div([
            html.H1("TRACLUS", className='main-title'),
        ], className='select-title'),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='experiment-dropdown',
                    options=[{'label': folder, 'value': folder} for folder in list_experiment_folders()],
                    placeholder="Selecciona una carpeta de experimento",
                    className='dropdown-experiment'
                ),
                dbc.Button('Acceder a experimentos anteriores', id='previous-exp-button', n_clicks=0, className='button1'), 
            ],className='button1-container'),

            html.Div([
                dbc.Button('Crear nuevo experimento', id='new-exp-button', n_clicks=0, className='button2')
            ], className='button2-container')

        ], className='buttons-wrapper'),

    ], className='grid-select-container')