from scrapy.selector import Selector


def get_texts(element: Selector):
    texts = (clean_text(text) for text in element.xpath(".//text()").getall())
    return list(filter(None, texts))  # remove empty texts


def clean_text(text: str):
    return text.strip().replace("\xa0", " ")
