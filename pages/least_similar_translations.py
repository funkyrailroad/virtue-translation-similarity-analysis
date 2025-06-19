import dash
import dash_bootstrap_components as dbc

from dash_data import least_similar_translations
from utils import dashify_dataframe

app = dash.get_app()

dash.register_page(__name__)


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
        dash.html.H2(
            children="Least Similar Translations",
            style={
                "textAlign": "center",
                "fontFamily": "Georgia, serif",
                "marginTop": "10px",
                "marginBottom": "20px",
                "borderBottom": "1px solid #ccc",
                "paddingBottom": "10px",
                "color": "#333",
            },
        ),
        least_similar_data_table,
    ]
)
