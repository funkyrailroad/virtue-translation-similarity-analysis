import dash_bootstrap_components as dbc
import dash
from dash import dcc, html
from flask import Flask


import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Flask(__name__)
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.YETI,
    ],
    use_pages=True,
    suppress_callback_exceptions=True,
    server=server,
)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dcc.Link(
                "The Aims",
                href="/the-aims",
                className="nav-link",
            ),
        ),
        dbc.NavItem(
            dcc.Link(
                "The Objects of Inquiry",
                href="/the-objects-of-inquiry",
                className="nav-link",
            ),
        ),
        dbc.NavItem(
            dcc.Link(
                "The Means",
                href="/the-means",
                className="nav-link",
            ),
        ),
        dbc.NavItem(
            dcc.Link(
                "The Ends",
                href="/the-ends",
                className="nav-link",
            ),
        ),
        # dbc.NavItem(
        #     dcc.Link(
        #         "Most Similar Translations",
        #         href="/most-similar-translations",
        #         className="nav-link",
        #     ),
        # ),
        # dbc.NavItem(
        #     dcc.Link(
        #         "Least Similar Translations",
        #         href="/least-similar-translations",
        #         className="nav-link",
        #     ),
        # ),
        # dbc.NavItem(
        #     dcc.Link(
        #         "Dimensionality Reduction",
        #         href="/dimensionality-reduction",
        #         className="nav-link",
        #     ),
        # ),
    ],
    brand="Home",
    brand_href="/",
)

app.layout = html.Div(
    children=[
        dcc.Location(id="url"),
        navbar,
        html.Div(dash.page_container),
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
