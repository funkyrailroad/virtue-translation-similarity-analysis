import dash
from dash import html

dash.register_page(__name__, path="/")

layout = html.Div(
    [
        html.Div(
            """
This might be a good spot for an introduction of the subject matter. Maybe even
list out all the quotes.
            """
        ),
    ]
)
