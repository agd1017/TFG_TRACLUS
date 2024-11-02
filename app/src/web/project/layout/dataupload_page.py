import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')

# Pagina Carga de datos

def get_page_dataUpdate():
    return html.Div([
        html.Div([
            html.H1("Introducción de datos previos"),
        ], className='box title'),
        html.Div([
            html.Div([
                html.H3("Introduce el nombre del experimento:"),
                dcc.Input(
                    id='input-name',
                    type='text',
                    placeholder='Nombre del experimento',
                    className='name-input'
                ),
            ], className='box inputtext'),
            html.Div([
                html.H3("Introduce el enlace del archivo que se va a analizar:"),
                dcc.Input(
                    id='input-url',
                    type='text',
                    placeholder='Introduce el enlace del archivo .csv',
                    className='file-upload'
                ),
            ], className='box inputfile'),
            html.Div([
                html.H3("Número de trayectorias que se van a usar:"),
                dcc.Input(
                    id='nrows-input',
                    type='number',
                    placeholder='Número de filas',
                    value='',
                    className='number-input'
                )
            ], className='box inputnumber'),
            html.Div([
                dbc.Button('Comenzar procesamiento', id='process-url-button', n_clicks=0)
            ], className='box buttonsconfirm'),
            html.Div([
                dbc.Button('Configuración predeterminada', id='default-config-button', n_clicks=0)
            ], className='box buttonsdefault'),
        ], className='grid-data-container'),
        html.Div([
            dbc.Spinner(children=[
                html.Div(id='output-container', className='box output')
            ])
        ], className='box output'),
        html.Div([
            dcc.Store(id='data-store')
        ], className='box data-store')        
    ], className='gid-dataupdate-container')