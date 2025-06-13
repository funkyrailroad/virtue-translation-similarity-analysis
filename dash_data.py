import numpy as np
import pandas as pd
import plotly.graph_objects as go
from textwrap3 import wrap
from data import translations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging


# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wrap_text(text, width=50):
    return "<br>".join(wrap(text, width=width))


@lru_cache(maxsize=5)
def get_model():
    """Load and return the SentenceTransformer model."""
    return SentenceTransformer(model_name)


@lru_cache(maxsize=5)
def vectorize_translations(translation_texts) -> np.ndarray:
    logger.info("Vectorizing translations...")
    model = get_model()
    return model.encode(translation_texts)


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
translation_vectors = vectorize_translations(translation_texts)
for translation, vector in zip(translations, translation_vectors):
    translation["vector"] = vector


fig, cos_sim_matrix = calculate_and_display_cosine_similarity_matrix(
    translation_texts,
    translation_vectors,
    cutoff=0.5,
)
