from dash import dcc, html, register_page
import dash_bootstrap_components as dbc
from written_word import home_text

register_page(__name__, path="/")

layout = dbc.Container(
    [
        html.H1(
            "Investigating various translations of passages",
        ),
        dcc.Markdown(home_text),
    ]
)
