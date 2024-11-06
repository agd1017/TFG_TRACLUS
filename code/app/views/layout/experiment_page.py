import dash_bootstrap_components as dbc
from dash import *
import matplotlib
matplotlib.use('Agg')

def experiment_row_OPTICS():
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
        dcc.Dropdown(
            id='dropdown-OPTICS-metric',
            options=[
                {'label': 'euclidean', 'value': 'euclidean'},
                {'label': 'l1', 'value': 'l1'},
                {'label': 'l2', 'value': 'l2'},
                {'label': 'manhattan', 'value': 'manhattan'},
                {'label': 'cosine', 'value': 'cosine'},
                {'label': 'cityblock', 'value': 'cityblock'}
            ],
            placeholder='Seleccionar metric',
            className='custom-dropdown'
        ),
        # Dropdown algorithm
        dcc.Dropdown(
            id='dropdown-OPTICS-algorithm',
            options=[
                {'label': 'auto', 'value': 'auto'},
                {'label': 'ball_tree', 'value': 'ball_tree'},
                {'label': 'kd_tree', 'value': 'kd_tree'},
                {'label': 'brute', 'value': 'brute'}
            ],
            placeholder='Seleccionar algorithm',
            className='custom-dropdown'
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

def experiment_row_DBSCAN():
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
        dcc.Dropdown(
            id='dropdown-DBSCAN-metric',
            options=[
                {'label': 'euclidean', 'value': 'euclidean'},
                {'label': 'l1', 'value': 'l1'},
                {'label': 'l2', 'value': 'l2'},
                {'label': 'manhattan', 'value': 'manhattan'},
                {'label': 'cosine', 'value': 'cosine'},
                {'label': 'cityblock', 'value': 'cityblock'}
            ],
            placeholder='Seleccionar metric',
            className='custom-dropdown'
        ),
        # Dropdown algorithm
        dcc.Dropdown(
            id='dropdown-DBSCAN-algorithm',
            options=[
                {'label': 'auto', 'value': 'auto'},
                {'label': 'ball_tree', 'value': 'ball_tree'},
                {'label': 'kd_tree', 'value': 'kd_tree'},
                {'label': 'brute', 'value': 'brute'}
            ],
            placeholder='Seleccionar algorithm',
            className='custom-dropdown'
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

def experiment_row_HDBSCAN():
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
        dcc.Dropdown(
            id='dropdown-HDBSCAN-metric',
            options=[
                {'label': 'euclidean', 'value': 'euclidean'},
                {'label': 'l1', 'value': 'l1'},
                {'label': 'l2', 'value': 'l2'},
                {'label': 'manhattan', 'value': 'manhattan'},
                {'label': 'cosine', 'value': 'cosine'},
                {'label': 'cityblock', 'value': 'cityblock'}
            ],
            placeholder='Seleccionar metric',
            className='custom-dropdown'
        ),
        # Dropdown algorithm
        dcc.Dropdown(
            id='dropdown-HDBSCAN-algorithm',
            options=[
                {'label': 'auto', 'value': 'auto'},
                {'label': 'ball_tree', 'value': 'ball_tree'},
                {'label': 'kd_tree', 'value': 'kd_tree'},
                {'label': 'brute', 'value': 'brute'}
            ],
            placeholder='Seleccionar algorithm',
            className='custom-dropdown'
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

def experiment_row_Aggl():
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
        dcc.Dropdown(
            id='dropdown-AgglomerativeClustering-metric',
            options=[
                {'label': 'euclidean', 'value': 'euclidean'},
                {'label': 'l1', 'value': 'l1'},
                {'label': 'l2', 'value': 'l2'},
                {'label': 'manhattan', 'value': 'manhattan'},
                {'label': 'cosine', 'value': 'cosine'},
                {'label': 'cityblock', 'value': 'cityblock'}
            ],
            placeholder='Seleccionar metric',
            className='custom-dropdown'
        ),
        # Dropdown linkage
        dcc.Dropdown(
            id='dropdown-AgglomerativeClustering-linkage',
            options=[
                {'label': 'ward', 'value': 'ward'},
                {'label': 'complete', 'value': 'complete'},
                {'label': 'average', 'value': 'average'},
                {'label': 'single', 'value': 'single'}
            ],
            placeholder='Seleccionar linkage',
            className='custom-dropdown'
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

def experiment_row_Spect():
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
        dcc.Dropdown(
            id='dropdown-SpectralClustering-affinity',
            options=[
                {'label': 'nearest_neighbors', 'value': 'nearest_neighbors'},
                {'label': 'rbf', 'value': 'rbf'},
                {'label': 'precomputed', 'value': 'precomputed'},
                {'label': 'precomputed_nearest_neighbors', 'value': 'precomputed_nearest_neighbors'}
            ],
            placeholder='Seleccionar affinity',
            className='custom-dropdown'
        ),
        # Dropdown assign_labels
        dcc.Dropdown(
            id='dropdown-SpectralClustering-assign_labels',
            options=[
                {'label': 'kmeans', 'value': 'kmeans'},
                {'label': 'discretize', 'value': 'discretize'},
                {'label': 'cluster_qr', 'value': 'cluster_qr'}
            ],
            placeholder='Seleccionar assign_labels',
            className='custom-dropdown'
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
        ], className='experiment-title'),
        # Contenedor con grid
        html.Div([
            html.Hr(),
            experiment_row_OPTICS(),
            html.Hr(),
            experiment_row_DBSCAN(),
            html.Hr(),
            experiment_row_HDBSCAN(),
            html.Hr(),
            experiment_row_Aggl(),
            html.Hr(),
            experiment_row_Spect(),
            html.Hr()
        ], className='grid-selecetors-container'),

        # Botón al final
        html.Div([
            dbc.Button('Ejecutar funciones', id='execute-button', color='primary', className='execute-button')
        ], className='button-container'),
    ], className='gid-experiment-container')


