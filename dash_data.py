from data import translations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.manifold import MDS
from textwrap3 import wrap
from tqdm import tqdm
import hashlib
import joblib
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


# get the row with min cosine similarity for each group
min_cos_sim = crossed[["quote_id_x", "cos_sim"]].groupby("quote_id_x").transform("min")


least_similar_translations = crossed[crossed.cos_sim == min_cos_sim.cos_sim]


translation_vectors_norm = np.sqrt((translation_vectors**2).sum(axis=1))
normed_translation_vectors = (
    translation_vectors / np.expand_dims(translation_vectors_norm, axis=0).transpose()
)

# MDS
n_components = 2
mds = MDS(n_components=n_components)

X_reduced = mds.fit_transform(normed_translation_vectors)

x_col_name = "Dimension 1"
y_col_name = "Dimension 2"
mds_df = pd.concat(
    [
        pd.DataFrame(X_reduced),
        df[["quote_id", "book_id"]],
    ],
    axis=1,
).rename(
    columns={0: x_col_name, 1: y_col_name},
)
plot_mds_df = mds_df
# subset_inds = [0, 1, 2, 3, 4, 5, 6, 7]
# plot_mds_df = mds_df[df.quote_id.isin(subset_inds)]
x_margin = 0.05 * (plot_mds_df[x_col_name].max() - plot_mds_df[x_col_name].min())
y_margin = 0.05 * (plot_mds_df[y_col_name].max() - plot_mds_df[y_col_name].min())
plot_mds_df.loc[:, "quote_id"] = plot_mds_df.quote_id.astype(str)
mds_fig = go.Figure(
    px.scatter(
        plot_mds_df,
        x=x_col_name,
        y=y_col_name,
        color="quote_id",
        hover_data=["book_id"],
        range_x=[
            plot_mds_df[x_col_name].min() - x_margin,
            plot_mds_df[x_col_name].max() + x_margin,
        ],
        range_y=[
            plot_mds_df[y_col_name].min() - y_margin,
            plot_mds_df[y_col_name].max() + y_margin,
        ],
    )
)
mds_fig.update_layout(
    title={
        "text": "MDS Visualization of Translation Embeddings",
        "xanchor": "center",
        "x": 0.5,
    },
    font=dict(
        family="Arial",
        size=14,
        color="black",
    ),
)


# UMAP
import umap

reducer = umap.UMAP(
    metric="cosine",
    # n_neighbors=30,
    n_components=2,
)
embedding = reducer.fit_transform(normed_translation_vectors)

x_col_name = "Dimension 1"
y_col_name = "Dimension 2"
umap_df = pd.concat(
    [pd.DataFrame(embedding), df[["quote_id", "book_id"]]], axis=1
).rename(
    columns={0: x_col_name, 1: y_col_name},
)

x_margin = 0.05 * (umap_df[x_col_name].max() - umap_df[x_col_name].min())
y_margin = 0.05 * (umap_df[y_col_name].max() - umap_df[y_col_name].min())
umap_df.loc[:, "quote_id"] = umap_df.quote_id.astype(str)
umap_fig = go.Figure(
    px.scatter(
        umap_df,
        x=x_col_name,
        y=y_col_name,
        color="quote_id",
        hover_data=["book_id"],
        range_x=[
            umap_df[x_col_name].min() - x_margin,
            umap_df[x_col_name].max() + x_margin,
        ],
        range_y=[
            umap_df[y_col_name].min() - y_margin,
            umap_df[y_col_name].max() + y_margin,
        ],
    )
)
umap_fig.update_layout(
    title={
        "text": "UMAP Visualization of Translation Embeddings",
        "xanchor": "center",
        "x": 0.5,
    },
    font=dict(
        family="Arial",
        size=14,
        color="black",
    ),
)


# TSNE
from sklearn.manifold import TSNE

x_col_name = "Dimension 1"
y_col_name = "Dimension 2"
reducer = TSNE(
    n_components=2,
    learning_rate="auto",
    init="random",
    # perplexity=3,
    metric="cosine",
)
embedding = reducer.fit_transform(normed_translation_vectors)
dim_red_df = pd.concat(
    [
        pd.DataFrame(embedding),
        df[["quote_id", "book_id"]],
    ],
    axis=1,
).rename(
    columns={0: x_col_name, 1: y_col_name},
)
x_margin = 0.05 * (dim_red_df[x_col_name].max() - dim_red_df[x_col_name].min())
y_margin = 0.05 * (dim_red_df[y_col_name].max() - dim_red_df[y_col_name].min())
dim_red_df.loc[:, "quote_id"] = dim_red_df.quote_id.astype(str)
tsne_fig = go.Figure(
    px.scatter(
        dim_red_df,
        x=x_col_name,
        y=y_col_name,
        color="quote_id",
        hover_data=["book_id"],
        range_x=[
            dim_red_df[x_col_name].min() - x_margin,
            dim_red_df[x_col_name].max() + x_margin,
        ],
        range_y=[
            dim_red_df[y_col_name].min() - y_margin,
            dim_red_df[y_col_name].max() + y_margin,
        ],
    )
)
tsne_fig.update_layout(
    title={
        "text": "T-SNE Visualization of Translation Embeddings",
        "xanchor": "center",
        "x": 0.5,
    },
    font=dict(
        family="Arial",
        size=14,
        color="black",
    ),
)
