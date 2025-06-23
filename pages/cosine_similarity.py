from dash import html, dcc, Input, Output
import dash
import dash_bootstrap_components as dbc

from dash_data import (
    full_cos_sim_matrix,
    cos_sim_matrix_fig,
    customdata,
    display_cosine_similarity_matrix,
    model_name,
    translation_texts,
)


app = dash.get_app()

dash.register_page(__name__)


layout = dbc.Container(
    [
        dash.dcc.Markdown(f"Embedding model used: {model_name}"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Clicked X Translation"),
                                dbc.CardBody(
                                    html.P(
                                        id="x-translation",
                                    )
                                ),
                            ],
                            className="mb-2",
                        ),
                    ],
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Clicked Y Translation"),
                                dbc.CardBody(
                                    html.P(
                                        id="y-translation",
                                    )
                                ),
                            ]
                        ),
                    ],
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Cosine Similarity Heatmap"),
                            dbc.CardBody(
                                [
                                    dash.html.Center(id="cos-sim", className="my-2"),
                                    html.Div(
                                        [
                                            dbc.Label("Slider", html_for="slider"),
                                            dcc.Slider(
                                                id="slider",
                                                min=0,
                                                max=1,
                                                step=0.1,
                                                value=0.5,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dcc.Graph(
                                        id="heatmap",
                                        figure=cos_sim_matrix_fig,
                                        style={
                                            "width": "800px",
                                            "height": "800px",
                                            "margin": "0 auto",
                                        },
                                    ),
                                ]
                            ),
                        ]
                    )
                )
            ]
        ),
    ],
    className="p-3",
)


@app.callback(
    Output("cos-sim", "children"),
    Output("x-translation", "children"),
    Output("y-translation", "children"),
    Input("heatmap", "clickData"),
)
def display_click_data(clickData):
    if clickData:
        point = clickData["points"][0]
        x_val = point["x"]
        y_val = point["y"]
        cos_sim = full_cos_sim_matrix[y_val][x_val]
        x_translation = translation_texts[x_val]
        y_translation = translation_texts[y_val]
        return (
            f"Cosine similarity: {cos_sim:.3f}",
            x_translation,
            y_translation,
        )
    return "Click on a cell to see its data", "", ""


@app.callback(
    Output("heatmap", "figure"),
    Input("slider", "value"),
)
def adjust_cos_sim_cutoff(value):
    fig = display_cosine_similarity_matrix(
        customdata=customdata,
        cosine_similarity_matrix=full_cos_sim_matrix,
        cutoff=value,
    )
    return fig
