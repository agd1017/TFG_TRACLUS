import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from controllers.callbacks import register_upload_callbacks
import os

# Definición de la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder='./views/assets', 
                external_stylesheets=[dbc.themes.FLATLY])

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
    port = int(os.environ.get("PORT", 8080))  # Obtiene el puerto de la variable de entorno o usa el 8080
    app.run_server(debug=True, host='0.0.0.0', port=port)  # Usar host 0.0.0.0 y puerto dinámico
    # http://127.0.0.1:8050/ , host='127.0.0.1', port=8050
