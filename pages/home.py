import dash
from dash_data import df
from dash import dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

layout = dbc.Container(
    [
        html.Div(
            """
This might be a good spot for an introduction of the subject matter. Maybe even
list out all the quotes.
            """
        ),
    ]
)
