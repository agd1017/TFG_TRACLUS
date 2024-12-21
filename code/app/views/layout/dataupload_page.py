import dash_bootstrap_components as dbc
from dash import html, dcc

def get_page_dataupdate():
    """
    Creates the layout for the data upload page where users can input experiment details,
    upload a file, and specify the number of trajectories to analyze.

    Returns:
        html.Div: A div containing the page layout with input fields and buttons for data upload.
    """
    return html.Div([
        # Title Section
        html.Div([
            html.H1("Introducción de datos previos"),  # Title of the page
        ], className='box title'),

        # Input Fields Section (Experiment Name, File Upload, Number of Trajectories)
        html.Div([
            # Experiment Name Input
            html.Div([
                html.H3("Introduce el nombre del experimento:"),  # Header for the experiment name input
                dcc.Input(
                    id='input-name',  # ID of the input field
                    type='text',  # Type of input (text)
                    placeholder='Nombre del experimento',  # Placeholder text
                    className='name-input'  # CSS class for styling
                ),
            ], className='box inputtext'),  # CSS class for styling the input box

            # File Upload Input
            html.Div([
                html.H3("Introduce el enlace del archivo que se va a analizar:"),  # Header for file upload input
                dcc.Upload(  # Upload component from Dash
                    id='upload-data',  # ID of the upload component
                    children=html.Button('Seleccionar archivo'),  # Button for file selection
                    multiple=False,  # Set to False to allow only one file upload at a time
                    accept='.csv,.xlsx',  # Allow only Excel files
                ),
            ], className='box inputfile'),  # CSS class for styling the file upload box

            # Number of Trajectories Input
            html.Div([
                html.H3("Número de trayectorias que se van a usar:"),  # Header for number of rows input
                dcc.Input(
                    id='nrows-input',  # ID of the input field
                    type='number',  # Type of input (number)
                    placeholder='Número de filas',  # Placeholder text
                    value='',  # Default value for the input
                    className='number-input'  # CSS class for styling the input box
                )
            ], className='box inputnumber'),  # CSS class for styling the number input box

            # Start Processing Button
            html.Div([
                dbc.Button('Comenzar procesamiento', id='process-url-button', n_clicks=0)  # Button to trigger processing
            ], className='box buttonsconfirm')  # CSS class for styling the button
        ], className='grid-data-container'),  # Container for the grid layout

        # Output Section (Loading Spinner and Output Display)
        html.Div([
            dbc.Spinner(children=[  # Loading spinner while data is being processed
                html.Div(id='output-container', className='box output')  # Area to show results
            ])
        ], className='box output'),  # CSS class for styling the output section

        # Data Store Component (to store data for use in other callbacks)
        html.Div([
            dcc.Store(id='data-store')  # Dash Store component to keep data for later use
        ], className='box data-store')  # CSS class for styling the data store section
    ], className='gid-dataupdate-container')  # CSS class for overall page layout
