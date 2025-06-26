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
                    dbc.Card(
                        [
                            dbc.CardHeader("Cosine Similarity Heatmap"),
                            dbc.CardBody(
                                [
                                    dash.html.Center(id="cos-sim", className="my-2"),
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
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H6(
                                    "Cosine Similarity Range",
                                    className="mb-0",
                                )
                            ),
                            dbc.Col(
                                html.Div(
                                    id="range-slider-values",
                                    className="text-end text-muted",
                                ),
                                width="auto",
                            ),
                        ],
                        className="align-items-center mb-2",
                        justify="between",
                    ),
                    dcc.RangeSlider(
                        id="range-slider",
                        min=0,
                        max=1,
                        step=0.01,
                        value=[0.5, 1.0],
                        tooltip={"placement": "bottom", "always_visible": True},
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        persistence=True,
                    ),
                ]
            ),
            className="mb-4",
        ),
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
    Input("range-slider", "value"),
)
def adjust_cos_sim_cutoff_range(value):
    fig = display_cosine_similarity_matrix(
        customdata=customdata,
        cosine_similarity_matrix=full_cos_sim_matrix,
        lower_cutoff=value[0],
        upper_cutoff=value[1],
    )
    return fig


@app.callback(
    Output("range-slider-values", "children"),
    Input("range-slider", "value"),
)
def update_range_display(value):
    return f"{value[0]:.2f} – {value[1]:.2f}"
