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
                html.Div([
                    dbc.Button('Acceder a experimento anteriores', id='load-exp-button', n_clicks=0, className='button1-load'), 
                    dbc.Button('Eliminar a experimento anteriores', id='delete-exp-button', n_clicks=0, className='button1-delete')
                ] ,className='button1-select-container'),               
            ],className='button1-container'),

            html.Div([
                dbc.Button('Crear nuevo experimento', id='new-exp-button', n_clicks=0, className='button2')
            ], className='button2-container')

        ], className='buttons-wrapper'),

        # Modal de confirmación
        dbc.Modal(
            [
                dbc.ModalHeader("Confirmación"),
                dbc.ModalBody("¿Estás seguro de que deseas eliminar este experimento? Esta acción no se puede deshacer."),
                dbc.ModalFooter([
                    dbc.Button("Cancelar", id="cancel-delete-button", color="secondary"),
                    dbc.Button("Eliminar", id="confirm-delete-button", color="danger"),
                ]),
            ],
            id="delete-modal",
            centered=True,
            is_open=False,
        ),

    ], className='grid-select-container')