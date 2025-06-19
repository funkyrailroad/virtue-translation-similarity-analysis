import dash
import dash_core_components as dcc
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
