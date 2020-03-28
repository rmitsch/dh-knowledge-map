import requests
import json
import wikipedia
from wikipedia import DisambiguationError, PageError

API_BASE = "http://api.dbpedia-spotlight.org/en"
endpoint = "/annotate"


# Integrate texts w/ presentation of texts
def integrate_wiki(keyword):
    keyword_page = keyword["label"]
    wiki_summary = ""
    try:
        wiki_summary = wikipedia.summary(keyword_page)
    except (DisambiguationError, PageError):
        pass
    return wiki_summary, "https://en.wikipedia.org/wiki/{}".format(keyword_page), keyword_page


def find_related_wiki(keyword, elements):
    elements_nodes = [d["data"].get("label") for d in elements]
    try:
        keyword_page = keyword["label"]
        related = wikipedia.search(keyword_page)[1:]
        related = [label for label in related if label not in elements_nodes][0:3]
    except (DisambiguationError, PageError):
        return []
    return related  # list of related wikipedia articles


# TODO disambiguation not showing text