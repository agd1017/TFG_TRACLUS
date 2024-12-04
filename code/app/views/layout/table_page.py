import dash_bootstrap_components as dbc
from dash import dash_table, html, dcc

def get_table(tabla):
    """
    Creates an interactive table to display the given data.

    The function converts any non-serializable values (like geometry objects) to strings and returns
    a Dash DataTable to display the data with features such as filtering, sorting, and pagination.

    Args:
        tabla (pandas.DataFrame): The data to be displayed in the table.

    Returns:
        dash_table.DataTable: The interactive table to display the data.
    """
    # Convert non-serializable values (e.g., geometry) to strings
    if 'geometry' in tabla.columns:
        tabla['geometry'] = tabla['geometry'].apply(lambda geom: str(geom))

    return dash_table.DataTable(
        id='table',  # Unique ID for the table component
        columns=[{"name": i, "id": i} for i in tabla.columns],  # Create column names dynamically
        data=tabla.to_dict('records'),  # Convert the DataFrame into a format Dash can use
        filter_action='native',  # Allow filtering within the table
        sort_action='native',  # Allow sorting by columns
        page_action='native',  # Enable pagination
        page_size=10,  # Number of rows per page
        style_table={'overflowX': 'auto'},  # Style the table for horizontal scrolling if necessary
    )

def get_page_tables(optics_on, hdbscan_on, dbscan_on, spect_on, aggl_on):
    """
    Creates the layout for the table page, including dropdown menus for selecting 
    clustering algorithms and displaying the corresponding data in a table and bar chart.

    Args:
        optics_on (bool): Whether the OPTICS option is enabled.
        hdbscan_on (bool): Whether the HDBSCAN option is enabled.
        dbscan_on (bool): Whether the DBSCAN option is enabled.
        spect_on (bool): Whether the Spectral Clustering option is enabled.
        aggl_on (bool): Whether the Agglomerative Clustering option is enabled.

    Returns:
        html.Div: The layout for the table page with dropdown menus, table, and bar chart.
    """
    # List of dropdown menu items based on which clustering algorithms are enabled
    item_table = [
        dbc.DropdownMenuItem("OPTICS", id="table-1", disabled=not optics_on), 
        dbc.DropdownMenuItem("HDBSCAN", id="table-2", disabled=not hdbscan_on),
        dbc.DropdownMenuItem("DBSCAN", id="table-3", disabled=not dbscan_on),
        dbc.DropdownMenuItem("SpectralClustering", id="table-4", disabled=not spect_on),
        dbc.DropdownMenuItem("AgglomerativeClustering", id="table-5", disabled=not aggl_on)
    ]

    return html.Div([  # Main container for the page layout
        dbc.Container([  # Bootstrap container for centralizing content
            html.H1("Tabla Interactiva en Dash", className="text-center my-3"),  # Title of the page
            html.Div([  # Dropdown menu for selecting clustering algorithm
                dbc.DropdownMenu(
                    item_table, label="Algoritmo de la tabla", color="primary"
                )
            ], className="box menu1"),
            dcc.Store(id='stored-data'),  # Store component to hold data on the client side
            html.Div([  # Container for the table with a spinner while loading data
                dbc.Spinner(children=[html.Div(id='table-container')])  
            ], className="box map1"),
            html.Div([  # Dropdown menu for selecting clusters to display and the bar chart
                html.Label("Selecciona Clústeres a Mostrar:"),
                dcc.Dropdown(
                    id='cluster-selector',
                    options=[
                        {'label': 'OPTICS', 'value': 'optics', 'disabled': not optics_on},
                        {'label': 'HDBSCAN', 'value': 'hdbscan', 'disabled': not hdbscan_on},
                        {'label': 'DBSCAN', 'value': 'dbscan', 'disabled': not dbscan_on},
                        {'label': 'Spectral Clustering', 'value': 'spectral', 'disabled': not spect_on},
                        {'label': 'Agglomerative Clustering', 'value': 'agglomerative', 'disabled': not aggl_on}
                    ],
                    value=None  # Default to no selection
                ),
                dcc.Graph(id='cluster-bar-chart')  # Bar chart to visualize cluster distribution
            ])
        ])
    ])
