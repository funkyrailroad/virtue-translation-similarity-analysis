from dash import get_app, register_page, html
import dash_bootstrap_components as dbc

from dash_data import least_similar_translations
from utils import dashify_dataframe

app = get_app()

register_page(__name__)


least_similar_data_table = dashify_dataframe(
    least_similar_translations[
        [
            "quote_id_x",
            "book_id_x",
            "text_x",
            "text_y",
            "book_id_y",
            "cos_sim",
        ]
    ].sort_values("cos_sim", ascending=True)
)

layout = dbc.Container(
    [
        html.H1("Least similar translations"),
        least_similar_data_table,
    ]
)
