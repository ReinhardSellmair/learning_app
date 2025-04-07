import requests
import os
import pandas as pd

from config import WIKI_SEARCH_URL, N_WIKI_PAGES
from utils import remove_angle_brackets

def search_wiki_pages(search_query: str, number_of_results: int=N_WIKI_PAGES) -> dict:
    headers = {'User-Agent': os.getenv('WIKI_USER_AGENT')}
    parameters = {'q': search_query, 'limit': number_of_results}
    response = requests.get(WIKI_SEARCH_URL, headers=headers, params=parameters)

    page_info = response.json()['pages']

    # remove searchmatch from convert exceprt
    for page in page_info:
        page['excerpt'] = remove_angle_brackets(page['excerpt'])

    return page_info

def get_wiki_page_info_df(keywords, number_of_results=N_WIKI_PAGES):
    # get dataframe with wiki page information for keywords
    # search wiki articles
    wiki_pages = []
    page_ids = set()
    for keyword in keywords:
        pages = search_wiki_pages(keyword, number_of_results)
        for page in pages:
            if page['id'] not in page_ids:
                wiki_pages.append(page)
                page_ids.add(page['id'])

    # convert to dataframe            
    pages_df = pd.DataFrame(wiki_pages)
    pages_df['n_rejected'] = 0

    return pages_df