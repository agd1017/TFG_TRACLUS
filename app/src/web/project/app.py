import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from callbacks import register_upload_callbacks

# Definición de la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

app.server.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5 GB en bytes

# Definición del layout principal de la aplicación utilizando componentes de Dash
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='navbar-container', className="navbar-container"),
    html.Div(id='page-content', className="page-content")  
], className="grid-main-container")

# Registrar callbacks
register_upload_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
    # http://127.0.0.1:8050/