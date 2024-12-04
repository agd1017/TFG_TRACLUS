import dash_bootstrap_components as dbc
from dash import html, dcc

from utils.data_utils import list_experiment_folders

def get_page_select():
    """
    Creates the layout for the main page, which includes a dropdown 
    to select an experiment folder, buttons for accessing or deleting previous 
    experiments, and a modal for confirming experiment deletion.

    Returns:
        html.Div: A Dash layout for selecting and managing experiments.
    """
    return html.Div([  # Main container for the page layout
        # Title section
        html.Div([
            html.H1("TRACLUS", className='main-title'),
        ], className='select-title'),  # Title of the page
        
        html.Div([  # Container for buttons
            html.Div([  # Section for experiment selection and actions
                dcc.Dropdown(
                    id='experiment-dropdown',
                    options=[{'label': folder, 'value': folder} for folder in list_experiment_folders()],
                    placeholder="Selecciona una carpeta de experimento",
                    className='dropdown-experiment'
                ),
                html.Div([  # Container for action buttons
                    dbc.Button('Acceder a experimento anteriores', id='load-exp-button', n_clicks=0, className='button1-load'), 
                    dbc.Button('Eliminar a experimento anteriores', id='delete-exp-button', n_clicks=0, className='button1-delete')
                ] ,className='button1-select-container'),               
            ],className='button1-container'),

            html.Div([  # Section for creating a new experiment button
                dbc.Button('Crear nuevo experimento', id='new-exp-button', n_clicks=0, className='button2')
            ], className='button2-container')

        ], className='buttons-wrapper'),  # Wrapper for all buttons

        # Confirmation modal for deleting an experiment
        dbc.Modal(
            [
                dbc.ModalHeader("Confirmación"),
                dbc.ModalBody("¿Estás seguro de que deseas eliminar este experimento? Esta acción no se puede deshacer."),
                dbc.ModalFooter([  # Modal footer with action buttons
                    dbc.Button("Cancelar", id="cancel-delete-button", color="secondary"),
                    dbc.Button("Eliminar", id="confirm-delete-button", color="danger"),
                ]),
            ],
            id="delete-modal",  # Modal identifier
            centered=True,  # Center the modal on the screen
            is_open=False,  # Modal is initially closed
        ),

    ], className='grid-select-container')  # Main container for the page layout