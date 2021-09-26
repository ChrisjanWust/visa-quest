import lxml
from utils.ad_pruning import remove_ads


def load_html(path: str):
    with open(path) as f:
        return lxml.html.fromstring(f.read())


def test_ad_removal_dice():
    html = load_html("utils/example_html/dice.html")
    assert html.xpath("//*[contains(@class, 'sticky-ad')]")
    html = remove_ads(html)
    assert not html.xpath("//*[contains(@class, 'sticky-ad')]")


if __name__ == "__main__":
    test_ad_removal_dice()
