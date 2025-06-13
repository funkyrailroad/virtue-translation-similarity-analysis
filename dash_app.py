import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html

import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=True,
    suppress_callback_exceptions=True,
)
dash.register_page(__name__)


app.layout = html.Div(
    children=[
        dcc.Location(id="url"),
        # Optional: shared navbar
        html.Nav(
            [
                dcc.Link(
                    "Home",
                    href="/",
                    style={"marginRight": "15px"},
                ),
                dcc.Link(
                    "Cosine Similarity",
                    href="/cosine-similarity",
                    style={"marginRight": "15px"},
                ),
            ],
            style={
                "padding": "20px",
                "borderBottom": "1px solid #ccc",
            },
        ),
        dash.html.H1(
            children="Investigating various translations of quotes",
            style={
                "textAlign": "center",
                "fontSize": "30px",
                "fontFamily": "Georgia, serif",
                "marginTop": "10px",
                "marginBottom": "20px",
                "borderBottom": "1px solid #ccc",
                "paddingBottom": "10px",
                "color": "#333",
            },
        ),
        html.Div(dash.page_container),
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
