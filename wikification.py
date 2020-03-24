import requests
import json
import wikipedia

API_BASE = "http://api.dbpedia-spotlight.org/en"
endpoint = "/annotate"

# Call endpoints for annotate URIs
def get_wiki_annotation(text):
    params = {
        "text": text["label"],
        "confidence": 0
    }
    response = requests.get(API_BASE+endpoint, headers={"accept": "application/json"},  params=params).json()
    uris = []
    if "Resources" in response:
        for resource in response["Resources"]:
            uris.append(resource.get("@URI"))
        info = []
        for uri in uris:
            info.append(requests.get(uri))
        return info


# Integrate texts w/ presentation of texts
def integrate_wiki(keyword):
    keyword_page = keyword["label"]
    return wikipedia.summary(keyword_page)


def find_related_wiki(keyword):
    keyword_page = keyword["label"]
    related = wikipedia.search(keyword_page)[1:4]
    return related  # list of keywords


