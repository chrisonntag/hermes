from .base import DataSource
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime
import requests
from typing import List, Dict, Optional


class Pocket(DataSource):
    """
    Uses the export HTML file of Pocket in order to create the data dict.
    They're structured as li elements in a ul list that contains one link in the following form:

    <a href="<link to source>" time_added="1706897206" tags="augmenting,companion,digital,rag">Title</a>
    """
    def __init__(self, name):
        self._name = name

    def load_content(self, url: str) -> Optional[str]:
        """Retrieve the HTML content of a web page.

        Args:
            link (str): The URL of the web page.

        Returns:
            str or None: The HTML content of the web page if successful, otherwise None.
        """
        try:
            print("Requesting ", url)
            res = requests.get(url)
            if res.status_code == 200:
                return res.text
            else:
                print("Failed to load HTML file. Status code:", res.status_code)
                return None
        except requests.exceptions.RequestException as e:
            print("Error: ", e)
            return None

    def extract_documents(self, html_content: str, stop_after: int=3, follow_links: bool=False) -> List[Dict]:
        """Extract links from an HTML file and append the content of the link. 

        Pocket offers to export all saved links as a single HTML file, which is used by 
        this method. 

        Args:
            html_content (str): The HTML content of the export file.
            stop_after (int): Limit of extracted links
            follow_links (bool): Request HTML content of each link

        Returns:
            List[Dict]: A list of structured data from the export file appended
                        with the HTML content of the saved link. 
        """
        links = []
        soup = BeautifulSoup(html_content, 'html.parser')
        for num, a_tag in enumerate(soup.find_all('a'), start=1):
            if num > stop_after:
                break

            href: str = a_tag.get('href')
            if href:
                link_data = {
                    'title': a_tag.get_text(strip=True),
                    'url': href,
                    'content': '',
                    'time_added': datetime.datetime.fromtimestamp(int(a_tag['time_added'])),
                    'tags': list(filter(None, a_tag['tags'].split(',')))
                }

                if follow_links:
                    content: Optional[str] = self.load_content(href)
                    link_data['content'] = content

                links.append(link_data)
        return links

    def get_name(self) -> str:
        return self._name


