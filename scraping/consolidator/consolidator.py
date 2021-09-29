import pandas as pd
from lxml import etree
from ml.predict.predict import predict_probas
from scraping.consolidator.element_annotator import extract_elements


def consolidate(tree: etree.HTML):
    elements = extract_elements(tree)
    texts = [element["text"] for element in elements]
    probas_df = pd.DataFrame(predict_probas(texts))
    probas_df["text"] = texts

    description_rows = probas_df[
        probas_df.description > probas_df.description.max() * 0.99
    ]
    if not description_rows:
        return
    # use the last, "deepest" node's text - cleaner and better for our title-before-description rule
    description_row = description_rows.iloc[-1]
    description = description_row.text

    probas_df = probas_df[: description_row.name].copy()

    title = get_last_valid_text("title", probas_df)
    if not title:
        return

    company = get_last_valid_text("company", probas_df)
    salary = get_last_valid_text("salary", probas_df)
    location = get_last_valid_text("location", probas_df)

    return {
        "title": title,
        "description": description,
        "company": company,
        "salary": salary,
        "location": location,
    }


def get_last_valid_text(column: str, df: pd.DataFrame, min_conf=0.5):
    valid_rows = df[df[column] > 0.5]
    if not valid_rows.empty:
        return valid_rows.iloc[-1].text
