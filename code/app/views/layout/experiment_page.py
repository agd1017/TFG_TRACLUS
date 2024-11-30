import dash_bootstrap_components as dbc
from dash import html, dcc

def experiment_row_optics():
    return html.Div([
        # Selector
        dcc.Checklist(
            id='selector-optics',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('OPTICS', className='row-label'),
        # Dropdown metric
        dcc.Dropdown(
            id='dropdown-optics-metric',
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
            id='dropdown-optics-algorithm',
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
            id='input-optics-eps',
            type='number',
            placeholder='Max_eps',
            disabled=True, 
            className='input'
        ),
        # Input numérico Min_samples
        dcc.Input(
            id='input-optics-sample',
            type='number',
            placeholder='Min_samples',
            disabled=True,  
            className='input'
        )
    ], className='grid-row-container')

def experiment_row_dbscan():
    return html.Div([
        # Selector
        dcc.Checklist(
            id='selector-dbscan',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('DBSCAN', className='row-label'),
        # Dropdown metric
        dcc.Dropdown(
            id='dropdown-dbscan-metric',
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
            id='dropdown-dbscan-algorithm',
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
            id='input-dbscan-eps',
            type='number',
            placeholder='Eps',
            disabled=True, 
            className='input'
        ),
        # Input numérico Min_samples
        dcc.Input(
            id='input-dbscan-sample',
            type='number',
            placeholder='Min_samples',
            disabled=True,  
            className='input'
        )
    ], className='grid-row-container')

def experiment_row_hdbscan():
    return html.Div([
        # Selector
        dcc.Checklist(
            id='selector-hdbscan',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('HDBSCAN', className='row-label'),
        # Dropdown metric
        dcc.Dropdown(
            id='dropdown-hdbscan-metric',
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
            id='dropdown-hdbscan-algorithm',
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
            id='input-hdbscan-sample',
            type='number',
            placeholder='Min_samples',
            disabled=True,  
            className='input'
        )
    ], className='grid-row-container')

def experiment_row_aggl():
    return html.Div([
        # Selector
        dcc.Checklist(
            id='selector-agglomerativeclustering',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('AgglomerativeClustering', className='row-label'),
        # Dropdown metric
        dcc.Dropdown(
            id='dropdown-agglomerativeclustering-metric',
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
            id='dropdown-agglomerativeclustering-linkage',
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
            id='input-agglomerativeclustering-n_clusters',
            type='number',
            placeholder='n_clusters',
            disabled=True, 
            className='input'
        )
    ], className='grid-row-container')

def experiment_row_spect():
    return html.Div([
        # Selector
        dcc.Checklist(
            id='selector-spectralclustering',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('SpectralClustering', className='row-label'),
        # Dropdown affinity
        dcc.Dropdown(
            id='dropdown-spectralclustering-affinity',
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
            id='dropdown-spectralclustering-assign_labels',
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
            id='input-spectralclustering-n_clusters',
            type='number',
            placeholder='n_clusters',
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
            experiment_row_optics(),
            html.Hr(),
            experiment_row_dbscan(),
            html.Hr(),
            experiment_row_hdbscan(),
            html.Hr(),
            experiment_row_aggl(),
            html.Hr(),
            experiment_row_spect(),
            html.Hr()
        ], className='grid-selecetors-container'),

        # Botón al final
        html.Div([
            dbc.Button('Ejecutar funciones', id='execute-button', color='primary', className='execute-button')
        ], className='button-container'),
    ], className='gid-experiment-container')


