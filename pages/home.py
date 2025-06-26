import dash
from dash import dcc
import dash_bootstrap_components as dbc
from written_word import home_text

dash.register_page(__name__, path="/")

layout = dbc.Container(
    [
        dcc.Markdown(home_text),
    ]
)
