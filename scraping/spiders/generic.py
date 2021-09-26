import logging
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.link import Link
from scraping.utils.string import *
from scraping.utils.url_classifier import derive_page_type, PageType
import re
from scraping.utils import get_domain
from scrapy_splash import SplashJsonResponse, SplashTextResponse


class CleanedLinkExtractor(LinkExtractor):
    def extract_links(self, response):
        def links_sorted_len_desc(links):
            return sorted(links, key=lambda link: len(link.url), reverse=True)

        all_links = super().extract_links(response)
        job_links = (
            link for link in all_links if derive_page_type(link.url) == PageType.AD
        )
        yield from links_sorted_len_desc(job_links)
        list_links = (
            link for link in all_links if derive_page_type(link.url) == PageType.LIST
        )
        yield from links_sorted_len_desc(list_links)


class GenericSpider(CrawlSpider):
    name = "generic"
    rules = (
        Rule(
            CleanedLinkExtractor(),
            callback="parse_page",
            follow=True,
            process_request="use_splash",
        ),
    )

    def __init__(self, *args, start_url: str = None, **kwargs):
        if start_url:
            assert not kwargs.get("start_urls")
            kwargs["start_urls"] = [start_url]
        if kwargs.get("allowed_domains") and isinstance(kwargs["allowed_domains"], str):
            kwargs["allowed_domains"] = kwargs["allowed_domains"].split(",")
        if kwargs.get("auto_allowed_domains"):
            assert not kwargs.get("allowed_domains")
            assert kwargs.get("start_urls")
            kwargs["allowed_domains"] = [
                get_domain(start_url) for start_url in kwargs["start_urls"]
            ]

        super().__init__(*args, **kwargs)

    def _requests_to_follow(self, response):
        if not isinstance(
            response, (scrapy.http.HtmlResponse, SplashJsonResponse, SplashTextResponse)
        ):
            return
        seen = set()
        for n, rule in enumerate(self._rules):
            links = [
                lnk
                for lnk in rule.link_extractor.extract_links(response)
                if lnk not in seen
            ]
            if links and rule.process_links:
                links = rule.process_links(links)
            for link in links:
                seen.add(link)
                r = self._build_request(n, link)
                yield rule.process_request(r)

    def start_requests(self):
        return map(self.use_splash, super().start_requests())

    def parse_page(self, response: scrapy.http.Response):
        if derive_page_type(response.url) == PageType.AD:
            yield {
                "url": response.url,
                "title": response.css("title::text").get(),
            }

    @staticmethod
    def use_splash(request, *args):
        assert isinstance(request, scrapy.Request)
        request.meta["splash"] = {
            "endpoint": "render.html",
            "args": {
                "wait": 3,
            },
        }
        return request
