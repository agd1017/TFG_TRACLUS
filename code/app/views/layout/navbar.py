import dash_bootstrap_components as dbc
from dash import dcc

def get_navbar (pathname):
    disabled = pathname in ['/', '/new-experiment', '/data-update']

    # Generar la barra de navegación dinámicamente con los botones habilitados o deshabilitados
    return dbc.Navbar(
        dbc.Container(children=[
            dbc.NavItem(dbc.NavLink("TRACLUS", href="/", className="navbar-text-title", disabled=(pathname == '/'))),
            dbc.NavItem(dbc.NavLink("Mapa de trayectorias", href="/map-page", className="navbar-text", disabled=disabled)),
            dbc.NavItem(dbc.NavLink("Comparacion de algoritmos", href="/TRACLUS-map", className="navbar-text", disabled=disabled)),
            dbc.NavItem(dbc.NavLink("Estadísticas", href="/estadisticas", className="navbar-text", disabled=disabled)),
            # Botón de descarga de datos
            dbc.NavItem([
                dbc.Button("Descargar Datos", id="btn-download-txt", className="navbar-text", disabled=disabled),
                dcc.Download(id="download-text"),
            ])
        ]),
        color="success",
        className="header-navbar"
    )