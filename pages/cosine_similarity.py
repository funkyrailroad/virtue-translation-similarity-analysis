from dash import html, dcc, Input, Output
import dash
import dash_bootstrap_components as dbc

from dash_data import model_name, translation_texts, cos_sim_matrix, cos_sim_matrix_fig_sim_matrix_sim_matrix


app = dash.get_app()

dash.register_page(__name__)


layout = html.Div(
    [
        dash.html.Center(children=f"Embedding model used: {model_name}"),
        dash.html.Center(id="cos-sim", className="my-2"),
        dbc.Row(
            [
                dbc.Col(width=1),
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
                    width=5,
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
                    width=5,
                ),
                dbc.Col(width=1),
            ]
        ),
        dcc.Graph(
            id="heatmap",
            figure=cos_sim_matrix_fig_sim_matrix_sim_matrix,
            style={
                "width": "800px",
                "height": "800px",
                "margin": "0 auto",
            },
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
        cos_sim = cos_sim_matrix[y_val][x_val]
        x_translation = translation_texts[x_val]
        y_translation = translation_texts[y_val]
        return (
            f"Cosine similarity: {cos_sim:.3f}",
            x_translation,
            y_translation,
        )
    return "Click on a cell to see its data", "", ""
