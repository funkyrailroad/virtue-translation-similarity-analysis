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
    external_stylesheets=[
        dbc.themes.YETI,
    ],
    use_pages=True,
    suppress_callback_exceptions=True,
)
dash.register_page(__name__)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dcc.Link(
                "Cosine Similarity",
                href="/cosine-similarity",
                className="nav-link",
            ),
        ),
        dbc.NavItem(
            dcc.Link(
                "Most similar translations",
                href="/most-similar-translations",
                className="nav-link",
            ),
        ),
        dbc.NavItem(
            dcc.Link(
                "Least similar translations",
                href="/least-similar-translations",
                className="nav-link",
            ),
        ),
        dbc.NavItem(
            dcc.Link(
                "Dimensionality Reduction",
                href="/dimensionality-reduction",
                className="nav-link",
            ),
        ),
    ],
    brand="Home",
    brand_href="/",
)

app.layout = html.Div(
    children=[
        dcc.Location(id="url"),
        navbar,
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
