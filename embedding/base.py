from abc import ABC, abstractmethod
from typing import List
import requests


class BaseEmbedder(ABC):
    @abstractmethod
    def generate(self):
        pass


class HuggingFaceEmbedder(BaseEmbedder):
    """Embedding class that offers embedding generation. 

    Currently only supporting HF public inference endpoints, but 
    this will soon change to local models via llamafile, therefore the class structure. 
    """
    def __init__(self, access_token: str, inference_url: str):
        self._token = access_token
        self._endpoint = inference_url

    def generate(self, text: str) -> List[float]:
        res = requests.post(self._endpoint, headers={"Authorization": f"Bearer {self._token}"}, json={"inputs": text})

        if res.status_code != 200:
            raise ValueError(f"Request failed with status code {res.status_code}: {res.text}")
        return res.json()
