
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc, State
import matplotlib
matplotlib.use('Agg')
from web_pages.Experiment_page import *

# Pagina Experimento

def get_page_select():
    return html.Div([
        # Título
        html.Div([
            html.H1("Seleccione una de las opciones"),
        ], className='title'),

        # Contenedor de botones, con flexbox para centrarlos horizontalmente
        html.Div([
            html.Div([
                dbc.Button('Acceder a experimentos anteriores', id='previous-exp-button', n_clicks=0, className='button')
            ], className='button1-container'),

            html.Div([
                dbc.Button('Crear nuevo experimento', id='new-exp-button', n_clicks=0, className='button')
            ], className='button2-container')
        ], className='buttons-wrapper')
    ], className='grid-select-container')














# Definición de la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

# Definición del layout principal de la aplicación utilizando componentes de Dash
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='navbar-container'),
    html.Div(id='page-content', className="page-content")  
], className="grid-main-container")

# Callbacks for nav bar

@app.callback(
    Output('navbar-container', 'children'),
    [Input('url', 'pathname')]
)
def update_navbar(pathname):
    disabledT = pathname == '/'
    disabled = pathname == '/' or pathname == '/new-experiment'

    # Generar la barra de navegación dinámicamente con los botones habilitados o deshabilitados
    navbar = dbc.Navbar(
        dbc.Container(children=[
            dbc.NavItem(dbc.NavLink("TRACLUS", href="/", className="navbar-text-title", disabled=disabledT)),
            dbc.NavItem(dbc.NavLink("Mapa de trayectorias", href="/home", className="navbar-text", disabled=disabled)),
            dbc.NavItem(dbc.NavLink("Comparacion de algoritmos", href="/comparacion", className="navbar-text", disabled=disabled)),
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
    
    return navbar

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return get_page_select()
    elif pathname == '/new-experiment':
        return get_page_experiment()
    else:
        return get_page_select()

# Callbacks for select page

@app.callback(
    Output('url', 'pathname'),
    [Input('previous-exp-button', 'n_clicks'),
    Input('new-exp-button', 'n_clicks')]
)
def navigate_to_page(n_clicks_previous, n_clicks_new):
    if n_clicks_previous > 0:
        return '/previous-experiments'
    elif n_clicks_new > 0:
        return '/new-experiment'
    return '/'

# Callbacks for experiment page

# Callbacks para habilitar/deshabilitar los elementos en función de los selectores
for i in range(1, 6):
    @app.callback(
        [Output(f'dropdown-{i}', 'disabled'),
        Output(f'input-{i}', 'disabled')],
        [Input(f'selector-{i}', 'value')]
    )
    def toggle_row_controls(selector_value):
        is_enabled = 'on' in selector_value
        return not is_enabled, not is_enabled  # Invertimos para habilitar/deshabilitar





































if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
    # http://127.0.0.1:8050/