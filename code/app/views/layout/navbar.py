import dash_bootstrap_components as dbc
from dash import dcc

def get_navbar(pathname):
    """
    Creates the navigation bar layout, where each item is enabled or disabled based on the current pathname.
    It includes links to different pages of the app and a button for downloading data.

    Args:
        pathname (str): The current URL path of the application. Used to determine which items should be disabled.

    Returns:
        dbc.Navbar: A Dash Bootstrap Navbar with dynamically enabled or disabled items based on the pathname.
    """
    
    # Disable navigation items based on the current pathname (e.g., home, new experiment, data update)
    disabled = pathname in ['/', '/new-experiment', '/data-update']

    # Generate the navbar with the appropriate buttons enabled or disabled
    return dbc.Navbar(
        dbc.Container(children=[
            # Logo and Home link
            dbc.NavItem(dbc.NavLink("TRACLUS", href="/", className="navbar-text-title", disabled=(pathname == '/'))),
            
            # Path to the map page, disabled based on current pathname
            dbc.NavItem(dbc.NavLink("Mapa de trayectorias", href="/map-page", className="navbar-text", disabled=disabled)),
            
            # Path to the algorithm comparison page, disabled based on current pathname
            dbc.NavItem(dbc.NavLink("Comparacion de algoritmos", href="/TRACLUS-map", className="navbar-text", disabled=disabled)),
            
            # Path to the statistics page, disabled based on current pathname
            dbc.NavItem(dbc.NavLink("Estadísticas", href="/estadisticas", className="navbar-text", disabled=disabled)),
            
            # Download data button, disabled based on current pathname
            dbc.NavItem([
                dbc.Button("Descargar Datos", id="btn-download-txt", className="btn btn-download", disabled=disabled),
                dcc.Download(id="download-text"),
            ])
        ]),
        color="success",  # Green color for the navbar background
        className="header-navbar"  # CSS class for styling the navbar
    )