import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc, State
import matplotlib
matplotlib.use('Agg')

# Pagina de experimento
def expirement_row(row_id):
    """Crea una fila con un selector, un dropdown y un input de número"""
    return html.Div([
        # Selector
        dcc.Checklist(
            id=f'selector-{row_id}',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div(f'Fila {row_id}', className='row-label'),
        # Dropdown
        dbc.DropdownMenu(
            label=f'Seleccionar opción {row_id}',
            id=f'dropdown-{row_id}',
            children=[
                dbc.DropdownMenuItem(f'Opción 1 - {row_id}', id=f'opcion-1-{row_id}'),
                dbc.DropdownMenuItem(f'Opción 2 - {row_id}', id=f'opcion-2-{row_id}')
            ],
            className='dropdown'
        ),
        # Input numérico
        dcc.Input(
            id=f'input-{row_id}',
            type='number',
            placeholder=f'Valor {row_id}',
            disabled=True,  # Inicialmente deshabilitado
            className='input'
        )
    ], className='grid-row-container')

def expirement_row_OPTICS():
    return html.Div([
        # Selector
        dcc.Checklist(
            id=f'selector-OPTICS',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div(f'OPTICS', className='row-label'),
        # Dropdown
        dbc.DropdownMenu(
            label=f'Seleccionar opción',
            id=f'dropdown-OPTICS',
            children=[
                dbc.DropdownMenuItem(f'Opción 1', id=f'opcion_1.1'),
                dbc.DropdownMenuItem(f'Opción 2', id=f'opcion_2.1')
            ],
            className='dropdown'
        ),
        # Input numérico
        dcc.Input(
            id=f'input-OPTICS-eps',
            type='number',
            placeholder=f'Max_eps',
            disabled=True, 
            className='input'
        ),
        # Input numérico
        dcc.Input(
            id=f'input-OPTICS-sample',
            type='number',
            placeholder=f'Min_samples',
            disabled=True,  
            className='input'
        )
    ], className='grid-row-container')

def get_page_experiment():
    return html.Div([
        # Título
        html.Div([
            html.H1("Configuración del experimento"),
        ], className='title'),
        # Contenedor con grid
        html.Div([
            html.Hr(),
            expirement_row(1),
            html.Hr(),
            expirement_row(2),
            html.Hr(),
            expirement_row(3),
            html.Hr(),
            expirement_row(4),
            html.Hr(),
            expirement_row(5),
            html.Hr()
        ], className='grid-selecetors-container'),

        # Botón al final
        html.Div([
            dbc.Button('Ejecutar funciones', id='execute-button', color='primary', className='execute-button')
        ], className='button-container'),
    ], className='gid-experiment-container')