import os
import joblib
import hashlib
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


def get_cache_filename(texts):
    joined = "".join(texts)
    h = hashlib.md5(joined.encode()).hexdigest()
    return f".vector_cache_{h}.joblib"


@lru_cache(maxsize=5)
def vectorize_translations(translation_texts) -> np.ndarray:
    logger.info("Vectorizing translations...")
    cache_file = get_cache_filename(translation_texts)
    if os.path.exists(cache_file):
        print(f"Loading vectors from {cache_file}")
        return joblib.load(cache_file)
    print(f"Saved vectors to {cache_file}")
    model = get_model()
    translation_vectors = model.encode(translation_texts)
    joblib.dump(translation_vectors, cache_file)
    return translation_vectors


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


cos_sim_matrix_fig, cos_sim_matrix = calculate_and_display_cosine_similarity_matrix(
    translation_texts,
    translation_vectors,
    cutoff=0.5,
)

df = pd.DataFrame(translations)
df = df.drop(columns=["id"]).reset_index(names="id")
print(df)

# prepare data from crossed columns
cross_cols = [
    "id",
    "text",
    "quote_id",
    "book_id",
    "vector",
]
crossed = df[cross_cols].merge(df[cross_cols], how="cross")[
    [
        "id_x",
        "id_y",
        "quote_id_x",
        "quote_id_y",
        "book_id_x",
        "book_id_y",
        "text_x",
        "text_y",
        "vector_x",
        "vector_y",
    ]
]

# remove comparisons of translations with themselves
crossed = crossed[crossed["id_x"] != crossed["id_y"]]

# remove comparisons of different quotes
crossed = crossed[crossed["quote_id_x"] == crossed["quote_id_y"]]

# remove inverse duplicates i.e. only have one of (a,b) and (b,a)
crossed["pair_key"] = crossed[["id_x", "id_y"]].apply(frozenset, axis=1)
crossed = crossed.drop_duplicates("pair_key").drop(columns=["pair_key"])

crossed["cos_sim"] = crossed.apply(
    lambda row: cosine_similarity(row["vector_x"], row["vector_y"]), axis=1
)

# confirm there are no comparisons across quotes, and quote_id_x == quote_id_y
(crossed[crossed["quote_id_x"] == crossed["quote_id_y"]] == crossed).all().all() == True

# get the row with max cosine similarity for each group
max_cos_sim = crossed[["quote_id_x", "cos_sim"]].groupby("quote_id_x").transform("max")

most_similar_translations = crossed[crossed["cos_sim"] == max_cos_sim["cos_sim"]]
most_similar_translations[
    ["quote_id_x", "book_id_x", "text_x", "text_y", "book_id_y", "cos_sim"]
].sort_values("cos_sim", ascending=True)


# get the row with min cosine similarity for each group
min_cos_sim = crossed[["quote_id_x", "cos_sim"]].groupby("quote_id_x").transform("min")
min_cos_sim


least_similar_translations = crossed[crossed.cos_sim == min_cos_sim.cos_sim]
least_similar_translations[
    ["quote_id_x", "book_id_x", "text_x", "text_y", "book_id_y", "cos_sim"]
].sort_values("cos_sim", ascending=False)
