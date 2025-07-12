from dash import dcc, html, register_page
import dash_bootstrap_components as dbc
from written_word import home_text

register_page(__name__)

layout = dbc.Container(
    [
        html.H1("The Aims"),
        dbc.Row(
            dbc.Col(
                dcc.Markdown(home_text),
                md="8",
            ),
            justify="center",
        ),
    ]
)
