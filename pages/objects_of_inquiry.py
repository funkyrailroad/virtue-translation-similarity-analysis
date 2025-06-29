from dash import get_app, register_page, html, dcc
import dash_bootstrap_components as dbc

from data import passages, translations
from written_word import dataset_explanation

app = get_app()

register_page(__name__)

# have whole flat denormalized representation of the translations, and use
# groupbys


def get_translation_texts_by_passage_id(quote_id, translations):
    return [html.P(t["text"]) for t in translations if t["quote_id"] == quote_id]


layout = dbc.Container(
    [
        html.H1("Objects of Inquiry"),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    get_translation_texts_by_passage_id(passage["id"], translations),
                    title=passage["description"],
                )
                for passage in passages
            ],
            start_collapsed=True,
        ),
    ]
)

"""
- maybe some dropdown menus
- accordions for the passages, which drop down into the translations
    - with a listgroup
        - https://www.dash-bootstrap-components.com/docs/components/list_group/
    - top
- possible use cases:
    - look at translations for a given passage
    - look at translations for a given passage from a selection of specific
      books
    - look at translations for a given passage from specific books
    - look at translations
- two columns:
    - left column can be the higher level data property (book, passage)
    - right column can be the lower level data property (translation)
- use pagination to avoid displaying all the data at once
    - https://www.dash-bootstrap-components.com/docs/components/pagination/
"""
