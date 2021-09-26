from lxml import etree
import urllib.parse
from scrapy.http.response import Response
from functools import reduce


def get_domain(url):
    return urllib.parse.urlparse(url).netloc


def clean_html(tree: Response):
    for element in tree.xpath("//style|//script"):
        remove_element(element)
    return tree


def remove_element(el: etree.Element):
    el.getparent().remove(el)


def pipe(*funcs):
    """
    Produces a function from a list of functions.
    Each function is called in order, starting with the initial argument
    and calling subsequent functions with their prior functions results.
    """

    def reducer(initial):
        return reduce(lambda data, func: func(data), funcs, initial)

    return reducer
