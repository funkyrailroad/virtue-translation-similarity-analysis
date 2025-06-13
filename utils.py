import dash_table
import pandas as pd


def dashify_dataframe(df: pd.DataFrame, page_size=10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in df.columns],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "5px",
            "minWidth": "100px",
            "width": "100px",
            "maxWidth": "180px",
            "whiteSpace": "normal",
        },
        # style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
        # filter_action="native",
        # sort_action="native",
        # export_format="csv",
        # export_headers="display",
        # style_data_conditional=[],
    )
