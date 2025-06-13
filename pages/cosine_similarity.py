from dash import html, dcc, Input, Output
from data import translations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from textwrap3 import wrap
from tqdm import tqdm
import dash
import dash_bootstrap_components as dbc
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.get_app()

dash.register_page(__name__)


def wrap_text(text, width=50):
    return "<br>".join(wrap(text, width=width))


@lru_cache(maxsize=5)
def get_model():
    """Load and return the SentenceTransformer model."""
    return SentenceTransformer(model_name)


@lru_cache(maxsize=5)
def vectorize_translations(translation_texts):
    logger.info("Vectorizing translations...")
    model = get_model()
    translation_vectors = model.encode(translation_texts)
    for translation, vector in zip(translations, translation_vectors):
        translation["vector"] = vector
    return translations


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_cosine_similarity_matrix(translation_texts, translation_vectors):
    similarities = {}
    scalar_grid = []

    for trait1, vector1 in tqdm(zip(translation_texts, translation_vectors)):
        scalar_row = []
        for trait2, vector2 in zip(translation_texts, translation_vectors):
            key = (trait1, trait2)
            similarity = cosine_similarity(vector1, vector2)
            similarities[key] = similarity
            scalar_row.append(similarity)
        scalar_grid.append(scalar_row)

    cosine_similarity_matrix = np.array(scalar_grid)
    return cosine_similarity_matrix


def display_cosine_similarity_matrix(
    labels,
    cosine_similarity_matrix,
    cutoff,
    char_limit=None,
):
    cosine_similarity_matrix[cosine_similarity_matrix < cutoff] = 0

    cdy = []
    for text in labels:
        row = []
        for _ in range(len(labels)):
            row.append(wrap_text(text[:char_limit]))
        cdy.append(row)

    cdx = []
    for _ in range(len(labels)):
        row = []
        for text in labels:
            row.append(wrap_text(text[:char_limit]))
        cdx.append(row)

    customdata = np.dstack((cdx, cdy))

    fig = go.Figure(
        data=go.Heatmap(
            z=cosine_similarity_matrix,
            zmin=0,
            zmax=1,
            customdata=customdata,
            # hovertemplate="<b>similarity:%{z:.3f}</b><br><br>x-text:%{customdata[0]}<br><br>y-text: %{customdata[1]}",
            hoverinfo="none",
        ),
    )
    fig.update_layout(
        title={
            "text": "Cosine Similarity Matrix",
            "xanchor": "center",
            "x": 0.5,
        },
        font=dict(family="Arial", size=14, color="black"),
        xaxis_title="X Translation",
        yaxis_title="Y Translation",
    )
    return fig


def calculate_and_display_cosine_similarity_matrix(
    translation_texts,
    translation_vectors,
    cutoff=0.5,
    char_limit=None,
):
    cosine_similarity_matrix = calculate_cosine_similarity_matrix(
        translation_texts,
        translation_vectors,
    )
    fig = display_cosine_similarity_matrix(
        translation_texts,
        cosine_similarity_matrix,
        cutoff,
        char_limit=char_limit,
    )
    return fig, cosine_similarity_matrix


model_name = "all-MiniLM-L6-v2"
translation_texts = tuple([d["text"] for d in translations])
translations = vectorize_translations(translation_texts)
translation_vectors = [d["vector"] for d in translations]


fig, cos_sim_matrix = calculate_and_display_cosine_similarity_matrix(
    translation_texts,
    translation_vectors,
    cutoff=0.5,
)
df = pd.DataFrame(translations).drop(columns=["vector"])

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
            figure=fig,
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
    # return "Click on a cell to see its data"
    return "Click on a cell to see its data", "", ""
