from dash import dcc, html, register_page
import dash_bootstrap_components as dbc
from written_word import home_text

register_page(__name__, path="/")

layout = dbc.Container(
    [
        html.H1(
            # "Comparing multiple translations of passages in the Nicomachean Ethics",
            "On the Many Ways to Skin a Cat: Translating an Ancient Text",
            # "Investigating various translations of passages",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Markdown(home_text),
                md="8",
            ),
            justify="center",
        ),
    ]
)
