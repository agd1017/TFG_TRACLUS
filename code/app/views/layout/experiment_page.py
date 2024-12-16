import dash_bootstrap_components as dbc
from dash import html, dcc

def experiment_row_optics():
    """
    This function generates the configuration row for the OPTICS clustering algorithm.
    It includes:
    - A checklist for enabling or disabling the OPTICS algorithm.
    - Dropdown menus to select the metric and algorithm for OPTICS.
    - Numeric input fields for setting the 'Max_eps' and 'Min_samples' parameters.
    """
    return html.Div([
        # Selector for enabling or disabling the OPTICS algorithm
        dcc.Checklist(
            id='selector-optics',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('OPTICS', className='row-label'),
        
        # Dropdown menu for selecting the metric used in OPTICS
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
        
        # Dropdown menu for selecting the algorithm to use in OPTICS
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
        
        # Numeric input field for setting 'Max_eps' parameter
        dcc.Input(
            id='input-optics-eps',
            type='number',
            placeholder='Max_eps',
            disabled=True,
            min=0,
            className='input'
        ),
        
        # Numeric input field for setting 'Min_samples' parameter
        dcc.Input(
            id='input-optics-sample',
            type='number',
            placeholder='Min_samples',
            disabled=True,
            min=0,
            className='input'
        )
    ], className='grid-row-container')


def experiment_row_dbscan():
    """
    This function generates the configuration row for the DBSCAN clustering algorithm.
    It includes:
    - A checklist for enabling or disabling the DBSCAN algorithm.
    - Dropdown menus to select the metric and algorithm for DBSCAN.
    - Numeric input fields for setting the 'Eps' and 'Min_samples' parameters.
    """
    return html.Div([
        # Selector for enabling or disabling the DBSCAN algorithm
        dcc.Checklist(
            id='selector-dbscan',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('DBSCAN', className='row-label'),
        
        # Dropdown menu for selecting the metric used in DBSCAN
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
        
        # Dropdown menu for selecting the algorithm to use in DBSCAN
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
        
        # Numeric input field for setting 'Eps' parameter
        dcc.Input(
            id='input-dbscan-eps',
            type='number',
            placeholder='Eps',
            disabled=True,
            min=0,
            className='input'
        ),
        
        # Numeric input field for setting 'Min_samples' parameter
        dcc.Input(
            id='input-dbscan-sample',
            type='number',
            placeholder='Min_samples',
            disabled=True, 
            min=0,
            className='input'
        )
    ], className='grid-row-container')


def experiment_row_hdbscan():
    """
    This function generates the configuration row for the HDBSCAN clustering algorithm.
    It includes:
    - A checklist for enabling or disabling the HDBSCAN algorithm.
    - Dropdown menus to select the metric and algorithm for HDBSCAN.
    - Numeric input fields for setting the 'Min_samples' parameter.
    """
    return html.Div([
        # Selector for enabling or disabling the HDBSCAN algorithm
        dcc.Checklist(
            id='selector-hdbscan',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('HDBSCAN', className='row-label'),
        
        # Dropdown menu for selecting the metric used in HDBSCAN
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
        
        # Dropdown menu for selecting the algorithm to use in HDBSCAN
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
        
        # Numeric input field for setting 'Min_samples' parameter
        dcc.Input(
            id='input-hdbscan-sample',
            type='number',
            placeholder='Min_samples',
            disabled=True, 
            min=0, 
            className='input'
        )
    ], className='grid-row-container')


def experiment_row_aggl():
    """
    This function generates the configuration row for the AgglomerativeClustering algorithm.
    It includes:
    - A checklist for enabling or disabling the AgglomerativeClustering algorithm.
    - Dropdown menus to select the metric and linkage method for AgglomerativeClustering.
    - Numeric input field for setting the 'n_clusters' parameter.
    """
    return html.Div([
        # Selector for enabling or disabling the AgglomerativeClustering algorithm
        dcc.Checklist(
            id='selector-agglomerativeclustering',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('AgglomerativeClustering', className='row-label'),
        
        # Dropdown menu for selecting the metric used in AgglomerativeClustering
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
        
        # Dropdown menu for selecting the linkage method in AgglomerativeClustering
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
        
        # Numeric input field for setting the 'n_clusters' parameter
        dcc.Input(
            id='input-agglomerativeclustering-n_clusters',
            type='number',
            placeholder='n_clusters',
            disabled=True, 
            min=0,
            className='input'
        )
    ], className='grid-row-container')


def experiment_row_spect():
    """
    This function generates the configuration row for the SpectralClustering algorithm.
    It includes:
    - A checklist for enabling or disabling the SpectralClustering algorithm.
    - Dropdown menus to select the affinity and assign_labels methods for SpectralClustering.
    - Numeric input field for setting the 'n_clusters' parameter.
    """
    return html.Div([
        # Selector for enabling or disabling the SpectralClustering algorithm
        dcc.Checklist(
            id='selector-spectralclustering',
            options=[{'label': '', 'value': 'on'}],
            value=[],
            className='selector'
        ),
        html.Div('SpectralClustering', className='row-label'),

        # Dropdown menu for selecting the affinity used in SpectralClustering
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

        # Dropdown menu for selecting the assign_labels method in SpectralClustering
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

        # Numeric input field for setting the 'n_clusters' parameter
        dcc.Input(
            id='input-spectralclustering-n_clusters',
            type='number',
            placeholder='n_clusters',
            disabled=True,  
            min=0,
            className='input'
        )
    ], className='grid-row-container')


def get_page_experiment():
    """
    This function generates the layout for the experiment configuration page.
    It includes:
    - A title for the page
    - A series of experiment configuration sections for different clustering methods
    - A button to trigger the execution of the experiment functions
    """
    return html.Div([

        # Title Section
        html.Div([
            html.H1("Configuración del experimento"),
        ], className='experiment-title'),

        # Configuration Rows for different clustering algorithms
        html.Div([
            html.Hr(),  # Horizontal rule for separation
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

        # Button to execute the selected experiment configurations
        html.Div([
            dbc.Button('Ejecutar funciones', id='execute-button', color='primary', className='execute-button')
        ], className='button-container'),

        # Modal for displaying the results of the experiment
        dbc.Modal(
            [
                dbc.ModalHeader("Falata seleccionar datos"),
                dbc.ModalBody("Como minimo selecione todos los datos de uno de los algoritmos."),
            ],
            id="exp-data-modal",
            centered=True,
            is_open=False
        )

    ], className='gid-experiment-container')


