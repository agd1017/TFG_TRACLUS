import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')
import os

# Pagina Experimento

EXPERIMENTS_DIR = "C:/Users/Álvaro/Documents/GitHub/TFG/TFG_TRACLUS/app/saved_results"

def list_experiment_folders():
    """Devuelve una lista con las carpetas dentro del directorio de resultados."""
    folders = [f for f in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, f))]
    return folders

def list_files_in_folder(folder_name):
    """Devuelve una lista de archivos dentro de la carpeta seleccionada."""
    folder_path = os.path.join(EXPERIMENTS_DIR, folder_name)
    files = os.listdir(folder_path)
    return files



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
        ], className='buttons-wrapper'),
        html.Div([
            html.H1("Selecciona un Experimento"),

            # Dropdown para seleccionar la carpeta de un experimento
            dcc.Dropdown(
                id='experiment-dropdown',
                options=[{'label': folder, 'value': folder} for folder in list_experiment_folders()],
                placeholder="Selecciona una carpeta de experimento"
            ),

        html.Div(id='file-list-container')
])


    ], className='grid-select-container')