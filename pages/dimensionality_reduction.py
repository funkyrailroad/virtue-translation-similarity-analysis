import dash
from dash import html, dcc

from dash_data import mds_fig, umap_fig, tsne_fig

app = dash.get_app()

dash.register_page(__name__)

layout = html.Div(
    [
        dcc.Graph(
            id="mds-figure",
            figure=mds_fig,
            style={
                "width": "800px",
                "height": "800px",
                "margin": "0 auto",
            },
        ),
        dcc.Graph(
            id="umap-figure",
            figure=umap_fig,
            style={
                "width": "800px",
                "height": "800px",
                "margin": "0 auto",
            },
        ),
        dcc.Graph(
            id="tsne-figure",
            figure=tsne_fig,
            style={
                "width": "800px",
                "height": "800px",
                "margin": "0 auto",
            },
        ),
    ]
)
