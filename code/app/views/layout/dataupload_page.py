import dash_bootstrap_components as dbc
from dash import html, dcc

def get_page_dataupdate():
    """
    Creates the layout for the data upload page with dynamic file name display.
    
    Returns:
        html.Div: A div containing the page layout.
    """
    return html.Div([
        # Title Section
        html.Div([
            html.H1("Introducción de datos previos")  # Title of the page
        ], className='title'),

        # Input Section
        html.Div([
            # Experiment Name Input
            html.Div([
                html.H3("Nombre del experimento:"),
                dcc.Input(
                    id='input-name',
                    type='text',
                    placeholder='Escribe el nombre del experimento',
                    className='name-input'
                ),
            ], className='inputtext'),

            # File Upload Input
            html.Div([
                html.H3("Selecciona el archivo a analizar:"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Arrastra el archivo o ',
                        html.A('haz clic aquí')
                    ], className='file-upload'),
                    multiple=False,
                    accept='.csv,.xlsx'
                ),
                # Area to display the selected file name
                html.Div(id='selected-file-name', className='selected-file')
            ], className='inputfile'),

            # Number of Trajectories Input
            html.Div([
                html.H3("Número de trayectorias:"),
                dcc.Input(
                    id='nrows-input',
                    type='number',
                    placeholder='Número de trayectorias',
                    className='number-input'
                ),
            ], className='inputnumber'),

            # Start Processing Button
            html.Div([
                dbc.Button(
                    'Comenzar procesamiento',
                    id='process-url-button',
                    n_clicks=0,
                    className='btn btn-confirm'
                )
            ], className='buttonsconfirm')
        ], className='grid-data-container'),

        # Output Section
        html.Div([
            dbc.Spinner([
                html.Div(id='output-container', className='output')  # Output display area
            ])
        ], className='output'),

        # Hidden Data Store
        dcc.Store(id='data-store')
    ], className='grid-dataupdate-container')

