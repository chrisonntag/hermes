from .base import DataSource
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime


class PocketSource(DataSource):
    """
    Uses the export HTML file of Pocket in order to create the data dict.
    They're structured as li elements in a ul list that contains one link in the following form:

    <a href="<link to source>" time_added="1706897206" tags="augmenting,companion,digital,rag">Title</a>
    """
    def __init__(self, name):
        self._name = name

    def get_documents(self, html_content):
        links = []
        soup = BeautifulSoup(html_content, 'html.parser')
        for a_tag in soup.find_all('a'):
            href = a_tag.get('href')
            if href:
                link_data = {
                    'title': a_tag.get_text(strip=True),
                    'url': href,
                    'time_added': datetime.datetime.fromtimestamp(int(a_tag['time_added'])),
                    'tags': list(filter(None, a_tag['tags'].split(',')))
                }
                links.append(link_data)
        return links

    def get_name(self):
        return self._name


