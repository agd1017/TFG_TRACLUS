import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')

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
        # Dropdown metric
        dbc.DropdownMenu(
            label=f'Seleccionar metric',
            id=f'dropdown-OPTICS-metric',
            children=[
                dbc.DropdownMenuItem(f'euclidean', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'l1', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'l2', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'manhattan', id=f'opcion_4'),
                dbc.DropdownMenuItem(f'cosine', id=f'opcion_5'),
                dbc.DropdownMenuItem(f'cityblock', id=f'opcion_6')
            ],
            className='dropdown'
        ),
        # Dropdown algorithm
        dbc.DropdownMenu(
            label=f'Seleccionar algorithm',
            id=f'dropdown-OPTICS-algorithm',
            children=[
                dbc.DropdownMenuItem(f'auto', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'ball_tree', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'kd_tree', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'brute', id=f'opcion_4')
            ],
            className='dropdown'
        ),
        # Input numérico Max_eps
        dcc.Input(
            id=f'input-OPTICS-eps',
            type='number',
            placeholder=f'Max_eps',
            disabled=True, 
            className='input'
        ),
        # Input numérico Min_samples
        dcc.Input(
            id=f'input-OPTICS-sample',
            type='number',
            placeholder=f'Min_samples',
            disabled=True,  
            className='input'
        )
    ], className='grid-row-container')

def expirement_row_DBSCAN():
    return html.Div([
        # Selector
        dcc.Checklist(
            id=f'selector-DBSCAN',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div(f'DBSCAN', className='row-label'),
        # Dropdown metric
        dbc.DropdownMenu(
            label=f'Seleccionar metric',
            id=f'dropdown-DBSCAN-metric',
            children=[
                dbc.DropdownMenuItem(f'euclidean', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'l1', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'l2', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'manhattan', id=f'opcion_4'),
                dbc.DropdownMenuItem(f'cosine', id=f'opcion_5'),
                dbc.DropdownMenuItem(f'cityblock', id=f'opcion_6')
            ],
            className='dropdown'
        ),
        # Dropdown algorithm
        dbc.DropdownMenu(
            label=f'Seleccionar algorithm',
            id=f'dropdown-DBSCAN-algorithm',
            children=[
                dbc.DropdownMenuItem(f'auto', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'ball_tree', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'kd_tree', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'brute', id=f'opcion_4')
            ],
            className='dropdown'
        ),
        # Input numérico Eps
        dcc.Input(
            id=f'input-DBSCAN-eps',
            type='number',
            placeholder=f'Eps',
            disabled=True, 
            className='input'
        ),
        # Input numérico Min_samples
        dcc.Input(
            id=f'input-DBSCAN-sample',
            type='number',
            placeholder=f'Min_samples',
            disabled=True,  
            className='input'
        )
    ], className='grid-row-container')

def expirement_row_HDBSCAN():
    return html.Div([
        # Selector
        dcc.Checklist(
            id=f'selector-HDBSCAN',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div(f'HDBSCAN', className='row-label'),
        # Dropdown metric
        dbc.DropdownMenu(
            label=f'Seleccionar metric',
            id=f'dropdown-HDBSCAN-metric',
            children=[
                dbc.DropdownMenuItem(f'euclidean', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'l1', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'l2', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'manhattan', id=f'opcion_4'),
                dbc.DropdownMenuItem(f'cosine', id=f'opcion_5'),
                dbc.DropdownMenuItem(f'cityblock', id=f'opcion_6')
            ],
            className='dropdown'
        ),
        # Dropdown algorithm
        dbc.DropdownMenu(
            label=f'Seleccionar algorithm',
            id=f'dropdown-HDBSCAN-algorithm',
            children=[
                dbc.DropdownMenuItem(f'auto', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'ball_tree', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'kd_tree', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'brute', id=f'opcion_4')
            ],
            className='dropdown'
        ),
        # Input numérico Min_samples
        dcc.Input(
            id=f'input-HDBSCAN-sample',
            type='number',
            placeholder=f'Min_samples',
            disabled=True,  
            className='input'
        )
    ], className='grid-row-container')

def expirement_row_AgglomerativeClustering():
    return html.Div([
        # Selector
        dcc.Checklist(
            id=f'selector-AgglomerativeClustering',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div(f'AgglomerativeClustering', className='row-label'),
        # Dropdown metric
        dbc.DropdownMenu(
            label=f'Seleccionar metric',
            id=f'dropdown-AgglomerativeClustering-metric',
            children=[
                dbc.DropdownMenuItem(f'euclidean', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'l1', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'l2', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'manhattan', id=f'opcion_4'),
                dbc.DropdownMenuItem(f'cosine', id=f'opcion_5'),
                dbc.DropdownMenuItem(f'cityblock', id=f'opcion_6')
            ],
            className='dropdown'
        ),
        # Dropdown linkage
        dbc.DropdownMenu(
            label=f'Seleccionar linkage',
            id=f'dropdown-AgglomerativeClustering-linkage',
            children=[
                dbc.DropdownMenuItem(f'ward', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'complete', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'average', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'single', id=f'opcion_4')
            ],
            className='dropdown'
        ),
        # Input numérico n_clusters
        dcc.Input(
            id=f'input-AgglomerativeClustering-n_clusters',
            type='number',
            placeholder=f'n_clusters',
            disabled=True, 
            className='input'
        )
    ], className='grid-row-container')

def expirement_row_SpectralClustering():
    return html.Div([
        # Selector
        dcc.Checklist(
            id=f'selector-SpectralClustering',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div(f'SpectralClustering', className='row-label'),
        # Dropdown affinity
        dbc.DropdownMenu(
            label=f'Seleccionar affinity',
            id=f'dropdown-SpectralClustering-affinity',
            children=[
                dbc.DropdownMenuItem(f'nearest_neighbors', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'rbf', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'precomputed', id=f'opcion_3'),
                dbc.DropdownMenuItem(f'precomputed_nearest_neighbors', id=f'opcion_4')
            ],
            className='dropdown'
        ),
        # Dropdown assign_labels
        dbc.DropdownMenu(
            label=f'Seleccionar assign_labels',
            id=f'dropdown-SpectralClustering-assign_labels',
            children=[
                dbc.DropdownMenuItem(f'kmeans', id=f'opcion_1'),
                dbc.DropdownMenuItem(f'discretize', id=f'opcion_2'),
                dbc.DropdownMenuItem(f'cluster_qr', id=f'opcion_3')
            ],
            className='dropdown'
        ),
        # Input numérico n_clusters
        dcc.Input(
            id=f'input-SpectralClustering-n_clusters',
            type='number',
            placeholder=f'n_clusters',
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
            expirement_row_OPTICS(),
            html.Hr(),
            expirement_row_DBSCAN(),
            html.Hr(),
            expirement_row_HDBSCAN(),
            html.Hr(),
            expirement_row_AgglomerativeClustering(),
            html.Hr(),
            expirement_row_SpectralClustering(),
            html.Hr()
        ], className='grid-selecetors-container'),

        # Botón al final
        html.Div([
            dbc.Button('Ejecutar funciones', id='execute-button', color='primary', className='execute-button')
        ], className='button-container'),
    ], className='gid-experiment-container')