# This script defines and runs the main Dash application. It includes:
# - Layout structure
# - Callbacks for navigation, UI interactivity, and data processing
# - Execution entry point

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import html, dcc
from controllers.callbacks import *
import os

# Create a Dash app instance with Bootstrap styling and custom asset folder
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,  # Avoid exceptions for unregistered callbacks
    assets_folder='./views/assets',  # Path to the assets folder
    external_stylesheets=[dbc.themes.FLATLY]  # Use Flatly theme from Bootstrap
)

# Set the maximum content length for file uploads to 5 GB
app.server.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024

# Layout definition
# Define the main structure of the application using Dash components
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Tracks the URL for navigation
    html.Div(id='navbar-container', className="navbar-container"),  # Container for the navigation bar
    html.Div(id='page-content', className="page-content")  # Container for dynamic page content
], className="grid-main-container")  # Main layout with a CSS class for grid styling

# --- Callbacks Section ---

@app.callback(
    Output('page-content', 'children'),  # Target: 'page-content' container
    [Input('url', 'pathname')]  # Trigger: Changes in the URL pathname
)
def callback_display_page(pathname):
    """
    Updates the main content of the application based on the current URL.

    Parameters:
        pathname (str): The current URL path.

    Returns:
        Dash component: The content to display on the page.
    """
    return display_page(pathname)

# -- Callbacks for navbar --

@app.callback(
    Output('navbar-container', 'children'),  # Target: 'navbar-container' container
    [Input('url', 'pathname')]  # Trigger: Changes in the URL pathname
)
def callback_update_navbar(pathname):
    """
    Updates the navigation bar dynamically based on the current URL.

    Parameters:
        pathname (str): The current URL path.

    Returns:
        Dash component: The updated navigation bar content.
    """
    return update_navbar(pathname)

@app.callback(
    Output("download-text", "data"),  # Target: Text file download data
    Input("btn-download-txt", "n_clicks"),  # Trigger: Click on download button
    prevent_initial_call=True  # Prevent callback from triggering on page load
)
def callback_download_data(n_clicks):
    """
    Generates and provides data for a text file download.

    Parameters:
        n_clicks (int): Number of clicks on the download button.

    Returns:
        data: Text data for the download.
    """
    return download_data(n_clicks)

# -- Callbacks for select page --

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),  # Target: URL pathname
    Input('new-exp-button', 'n_clicks'),  # Trigger: Clicks on "new experiment" button
    prevent_initial_call=True  # Prevent the callback from firing initially
)
def callback_navigate_experiment_page(n_clicks_new):
    """
    Navigates to the experiment page when the "new experiment" button is clicked.

    Parameters:
        n_clicks_new (int): Number of clicks on the button.

    Returns:
        str: The pathname for the experiment page.
    """
    return navigate_experiment_page(n_clicks_new)

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True), # Target: URL pathname
    Output('experiment-modal', 'is_open')], # Target: Modal visibility state
    [Input('experiment-dropdown', 'value'), # Trigger: Folder selection
    Input('load-exp-button', 'n_clicks')], # Trigger: Click on "load" button
    [State('experiment-modal', 'is_open')], # State: Current modal visibility
    prevent_initial_call=True # Prevent the callback from firing initially
)
def callback_display_files_in_selected_folder(folder_name, n_clicks_previous, is_modal_open):
    """
    Updates the application to display files from a selected folder.

    Parameters:
        folder_name (str): The name of the selected folder.
        n_clicks_previous (int): Number of clicks on the load button.
        is_modal_open (bool): Current state of the modal.

    Returns:
        str: The pathname for the map page.
        bool: The new state of the modal (open or closed).
    """
    return display_files_in_selected_folder(folder_name, n_clicks_previous, is_modal_open)

@app.callback(
    Output("delete-modal", "is_open"),  # Target: Modal's visibility state
    [Input("delete-exp-button", "n_clicks"),  # Trigger: Click on "delete" button
    Input("cancel-delete-button", "n_clicks"),  # Trigger: Click on "cancel" button
    Input("confirm-delete-button", "n_clicks")],  # Trigger: Click on "confirm" button
    [State("delete-modal", "is_open"),  # State: Current modal visibility
    State("experiment-dropdown", "value")],  # State: Selected folder
    prevent_initial_call=True
)
def callback_toggle_modal(n_delete, n_cancel, n_confirm, is_open, folder_name):
    """
    Toggles the delete confirmation modal's visibility.

    Parameters:
        n_delete (int): Number of clicks on the delete button.
        n_cancel (int): Number of clicks on the cancel button.
        n_confirm (int): Number of clicks on the confirm button.
        is_open (bool): Current state of the modal.
        folder_name (str): The selected folder to delete.

    Returns:
        bool: The new state of the modal (open or closed).
    """
    return toggle_modal(n_delete, n_cancel, n_confirm, is_open, folder_name)

@app.callback(
    Output("url", "pathname"),  # Target: URL pathname
    [Input("confirm-delete-button", "n_clicks")],  # Trigger: Click on "confirm delete" button
    [State("experiment-dropdown", "value")],  # State: Selected folder
    prevent_initial_call=True
)
def callback_delete_experiment(n_clicks, folder_name):
    """
    Deletes the selected experiment and updates the navigation.

    Parameters:
        n_clicks (int): Number of clicks on the confirm delete button.
        folder_name (str): The selected folder to delete.

    Returns:
        str: The pathname for the updated page after deletion.
    """
    return delete_experiment(n_clicks, folder_name)

# -- Callbacks for experiment page --

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),  # Target: URL pathname
    Output('exp-data-modal', 'is_open')],  # Target: Modal visibility state
    Input('execute-button', 'n_clicks'),  # Trigger: Click on execute button
    [
        # State variables for OPTICS algorithm configuration
        State('selector-optics', 'value'),
        State('dropdown-optics-metric', 'value'),
        State('dropdown-optics-algorithm', 'value'),
        State('input-optics-eps', 'value'),
        State('input-optics-sample', 'value'),
        # State variables for DBSCAN algorithm configuration
        State('selector-dbscan', 'value'),
        State('dropdown-dbscan-metric', 'value'),
        State('dropdown-dbscan-algorithm', 'value'),
        State('input-dbscan-eps', 'value'),
        State('input-dbscan-sample', 'value'),
        # State variables for HDBSCAN algorithm configuration
        State('selector-hdbscan', 'value'),
        State('dropdown-hdbscan-metric', 'value'),
        State('dropdown-hdbscan-algorithm', 'value'),
        State('input-hdbscan-sample', 'value'),
        # State variables for Agglomerative Clustering configuration
        State('selector-agglomerativeclustering', 'value'),
        State('dropdown-agglomerativeclustering-metric', 'value'),
        State('dropdown-agglomerativeclustering-linkage', 'value'),
        State('input-agglomerativeclustering-n_clusters', 'value'),
        # State variables for Spectral Clustering configuration
        State('selector-spectralclustering', 'value'),
        State('dropdown-spectralclustering-affinity', 'value'),
        State('dropdown-spectralclustering-assign_labels', 'value'),
        State('input-spectralclustering-n_clusters', 'value'),

        State('exp-data-modal', 'is_open')  # State: Modal visibility
    ],
    prevent_initial_call=True  # Prevent callback from triggering on page load
)
def callback_navigate_to_page_dataupdate(
    n_clicks_data, checkoptics, optics_metric_value, optics_algorithm_value,
    optics_eps_value, optics_sample_value, checkdbscan, dbscan_metric_value,
    dbscan_algorithm_value, dbscan_eps_value, dbscan_sample_value, checkhdbscan,
    hdbscan_metric_value, hdbscan_algorithm_value, hdbscan_sample_value,
    checkagglomerativeclustering, aggl_metric_value, aggl_linkage_value,
    aggl_n_clusters_value, checkspectralclustering, spect_affinity_value,
    spect_assign_labels_value, spect_n_clusters_value,
    is_open
):
    """
    Navigates to the data update page with selected clustering configurations.

    Parameters:
        n_clicks_data (int): Number of clicks on the execute button.
        checkoptics, checkdbscan, checkhdbscan, checkagglomerativeclustering, checkspectralclustering (bool): Flags for enabled clustering methods.
        optics_metric_value, dbscan_metric_value, hdbscan_metric_value, aggl_metric_value, spect_affinity_value (str): Selected metrics for each algorithm.
        optics_algorithm_value, dbscan_algorithm_value, hdbscan_algorithm_value, aggl_linkage_value, spect_assign_labels_value (str): Selected options for each algorithm.
        optics_eps_value, dbscan_eps_value (float): Epsilon values for clustering algorithms.
        optics_sample_value, dbscan_sample_value, hdbscan_sample_value (int): Sample size values for clustering.
        aggl_n_clusters_value, spect_n_clusters_value (int): Number of clusters for clustering algorithms.

    Returns:
        str: URL path to navigate to the data update page.
    """
    return navigate_to_page_dataupdate(
        n_clicks_data, checkoptics, optics_metric_value, optics_algorithm_value,
        optics_eps_value, optics_sample_value, checkdbscan, dbscan_metric_value,
        dbscan_algorithm_value, dbscan_eps_value, dbscan_sample_value, checkhdbscan,
        hdbscan_metric_value, hdbscan_algorithm_value, hdbscan_sample_value,
        checkagglomerativeclustering, aggl_metric_value, aggl_linkage_value,
        aggl_n_clusters_value, checkspectralclustering, spect_affinity_value,
        spect_assign_labels_value, spect_n_clusters_value,
        is_open
    )


# Callbacks to enable/disable controls based on selectors

@app.callback(
    [
        Output('dropdown-optics-metric', 'disabled'),
        Output('dropdown-optics-algorithm', 'disabled'),
        Output('input-optics-eps', 'disabled'),
        Output('input-optics-sample', 'disabled')
    ],
    [Input('selector-optics', 'value')]  # Trigger: Change in OPTICS selector
)
def callback_toggle_rowo_controls(selector_value_o):
    """
    Toggles the enabled/disabled state of OPTICS controls based on the selector value.

    Parameters:
        selector_value_o (bool): Whether OPTICS is enabled.

    Returns:
        list: Boolean values indicating control states (enabled/disabled).
    """
    return toggle_rowo_controls(selector_value_o)

@app.callback(
    [
        Output('dropdown-dbscan-metric', 'disabled'),
        Output('dropdown-dbscan-algorithm', 'disabled'),
        Output('input-dbscan-eps', 'disabled'),
        Output('input-dbscan-sample', 'disabled')
    ],
    [Input('selector-dbscan', 'value')]  # Trigger: Change in DBSCAN selector
)
def callback_toggle_rowd_controls(selector_value_d):
    """
    Toggles the enabled/disabled state of DBSCAN controls based on the selector value.

    Parameters:
        selector_value_d (bool): Whether DBSCAN is enabled.

    Returns:
        list: Boolean values indicating control states (enabled/disabled).
    """
    return toggle_rowd_controls(selector_value_d)

@app.callback(
    [
        Output('dropdown-hdbscan-metric', 'disabled'),
        Output('dropdown-hdbscan-algorithm', 'disabled'),
        Output('input-hdbscan-sample', 'disabled')
    ],
    [Input('selector-hdbscan', 'value')]  # Trigger: Change in HDBSCAN selector
)
def callback_toggle_rowh_controls(selector_value_h):
    """
    Toggles the enabled/disabled state of HDBSCAN controls based on the selector value.

    Parameters:
        selector_value_h (bool): Whether HDBSCAN is enabled.

    Returns:
        list: Boolean values indicating control states (enabled/disabled).
    """
    return toggle_rowh_controls(selector_value_h)

@app.callback(
    [
        Output('dropdown-agglomerativeclustering-metric', 'disabled'),
        Output('dropdown-agglomerativeclustering-linkage', 'disabled'),
        Output('input-agglomerativeclustering-n_clusters', 'disabled')
    ],
    [Input('selector-agglomerativeclustering', 'value')]  # Trigger: Change in Agglomerative Clustering selector
)
def callback_toggle_rowa_controls(selector_value_a):
    """
    Toggles the enabled/disabled state of Agglomerative Clustering controls based on the selector value.

    Parameters:
        selector_value_a (bool): Whether Agglomerative Clustering is enabled.

    Returns:
        list: Boolean values indicating control states (enabled/disabled).
    """
    return toggle_rowa_controls(selector_value_a)

@app.callback(
    [
        Output('dropdown-spectralclustering-affinity', 'disabled'),
        Output('dropdown-spectralclustering-assign_labels', 'disabled'),
        Output('input-spectralclustering-n_clusters', 'disabled')
    ],
    [Input('selector-spectralclustering', 'value')]  # Trigger: Change in Spectral Clustering selector
)
def callback_toggle_rows_controls(selector_value_s):
    """
    Toggles the enabled/disabled state of Spectral Clustering controls based on the selector value.

    Parameters:
        selector_value_s (bool): Whether Spectral Clustering is enabled.

    Returns:
        list: Boolean values indicating control states (enabled/disabled).
    """
    return toggle_rows_controls(selector_value_s)


# -- Callbacks for data upload page --

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),  # Target: URL pathname
    Output('output-container', 'children', allow_duplicate=True)],  # Output container
    [Input('process-url-button', 'n_clicks'),  # Trigger: Click on process button
    Input('upload-data', 'contents'),  # Uploaded data content
    Input('nrows-input', 'value'),  # Number of rows to process
    Input('input-name', 'value')],  # Folder name for the data
    prevent_initial_call=True  # Prevent callback from triggering on page load
)
def callback_process_csv_from_url(n_clicks_upload, data, nrows, folder_name):
    """
    Processes a CSV or EXCEL file uploaded by the user and navigates to the next page.

    Parameters:
        n_clicks_upload (int): Number of clicks on the process button.
        data (str): Content of the uploaded file.
        nrows (int): Number of rows to process from the file.
        folder_name (str): Name of the folder to save the processed data.

    Returns:
        tuple: URL path to navigate and a message for the output container.
    """
    return process_csv_from_url(n_clicks_upload, data, nrows, folder_name)

# -- Callbacks for map page --
    
@app.callback(
    Output('map-container', 'children'),  # Target: Map container content
    Input('url', 'pathname')  # Trigger: Change in URL pathname
)
def callback_update_map(*args):
    """
    Updates the map display based on the current URL path.

    Parameters:
        *args: Variable arguments for the map page.

    Returns:
        children: Updated content for the map container.
    """
    return update_map()

# -- Callbacks for TRACLUS map page --

@app.callback(
    Output('map-clusters-1', 'children'),  # Target: Cluster 1 map content
    [
        Input('item-1-1', 'n_clicks'),
        Input('item-1-2', 'n_clicks'),
        Input('item-1-3', 'n_clicks'),
        Input('item-1-4', 'n_clicks'),
        Input('item-1-5', 'n_clicks')
    ]  # Trigger: Clicks on clustering options
)
def callback_display_clusters_1(*args):
    """
    Displays cluster 1 map based on user interactions.

    Parameters:
        *args: Variable arguments for cluster selections.

    Returns:
        children: Content for the cluster 1 map container.
    """
    return display_clusters_1()

@app.callback(
    Output('map-clusters-2', 'children'),  # Target: Cluster 2 map content
    [
        Input('item-2-1', 'n_clicks'),
        Input('item-2-2', 'n_clicks'),
        Input('item-2-3', 'n_clicks'),
        Input('item-2-4', 'n_clicks'),
        Input('item-2-5', 'n_clicks')
    ]  # Trigger: Clicks on clustering options
)
def callback_display_clusters_2(*args):
    """
    Displays cluster 2 map based on user interactions.

    Parameters:
        *args: Variable arguments for cluster selections.

    Returns:
        children: Content for the cluster 2 map container.
    """
    return display_clusters_2()

# -- Callbacks for tables page --

@app.callback(
    Output('table-container', 'children'),  # Target: Table container content
    [
        Input('table-1', 'n_clicks'),
        Input('table-2', 'n_clicks'),
        Input('table-3', 'n_clicks'),
        Input('table-4', 'n_clicks'),
        Input('table-5', 'n_clicks')
    ]  # Trigger: Clicks on table options
)
def callback_update_table(*args):
    """
    Updates the displayed table based on user interactions.

    Parameters:
        *args: Variable arguments for table selections.

    Returns:
        children: Updated content for the table container.
    """
    return update_table()

@app.callback(
    Output('cluster-bar-chart', 'figure'),  # Target: Bar chart figure
    Input('cluster-selector', 'value')  # Trigger: Change in cluster selector value
)
def callback_update_graph(selected_filter):
    """
    Updates the cluster bar chart based on the selected filter.

    Parameters:
        selected_filter (str): Filter selected by the user.

    Returns:
        figure: Updated bar chart figure.
    """
    return update_graph(selected_filter)

# Entry point of the program
if __name__ == '__main__':
    # Retrieve the port number from environment variables or default to 8080
    port = int(os.environ.get("PORT", 8080))
    # Run the server on host '0.0.0.0' (accessible externally) and specified port
    app.run_server(host='0.0.0.0', port=port)
    # app.run_server( host='127.0.0.1', port=8050) 
