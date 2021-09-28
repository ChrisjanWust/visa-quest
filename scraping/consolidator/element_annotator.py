from lxml import etree
from scrapy.http.response import Response
from scrapy.http.response.html import HtmlResponse

from scraping.utils import clean_html, pipe


def get_text(el: etree.Element):
    return "\n".join(el.xpath(".//text()")).strip()


def extract_elements(tree: etree.HTML):
    tree = clean_html(tree)
    all_elements = list(tree.iter())
    previous_text = ""

    for el in tree.xpath("//*[text()!='']"):
        el: etree.Element
        text = get_text(el)

        if not text or el.tag in [
            "html",
            "body",
            "head",
            "button",
            "header",
            "nav",
            "footer",
        ]:
            continue

        yield {
            # "position": res_string.index(etree.tostring(el)) / len(res_string),
            "position": all_elements.index(el),
            # "position_frac": ... / len(all_elements),
            "text": text,
            "previous_text": previous_text,
            "nr_ancestors": len(list(el.iterancestors())),
            "nr_children": len(list(el.iterchildren())),
            "nr_descendants": len(list(el.iterdescendants())),
            "tag": el.tag,
            "child_tags": [child.tag for child in el.iterchildren() if child],
        }
        if text != previous_text:
            previous_text = text


def absolute_reverse(iterable):
    return reversed(list(iterable))


def level_of_mutual_parent(el_1: etree.Element, el_2: etree.Element):
    i = 0
    for parent_1, parent_2 in zip(
        absolute_reverse(el_1.iterancestors()), absolute_reverse(el_2.iterancestors())
    ):
        if parent_1 != parent_2:
            return i
        i += 1


def normalise_features(features):
    pass


def html_to_response(text: str) -> Response:
    return HtmlResponse(url="", body=text, encoding="utf-8")


#%%
def extract_elements_from_text(html_text):
    return pipe(
        etree.HTML,
        extract_elements,
        list,
    )(html_text)


if __name__ == "__main__":
    from pprint import pprint

    with open("tmp/executiveplacements.html", "r") as f:
        text = f.read()

    elements = extract_elements_from_text(text)
    pprint(elements)
