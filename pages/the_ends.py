from dash import html, dcc
import dash
import dash_bootstrap_components as dbc

from written_word import conclusion_text


app = dash.get_app()

dash.register_page(__name__)


layout = dbc.Container(
    [
        html.H1("The Ends"),
        dbc.Row(
            dbc.Col(
                [
                    dcc.Markdown(conclusion_text),
                ],
                md="8",
            ),
            justify="center",
        ),
    ]
)
