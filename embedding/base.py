import requests
from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self):
        pass


class HuggingFaceEmbedder(BaseEmbedder):
    """Embedding class that offers embedding generation. 

    Currently only supporting HF public inference endpoints, but 
    this will soon change to local models via llamafile, therefore the class structure. 
    """
    def __init__(self, access_token: str, inference_url: str):
        self._token = access_token
        self._endpoint = inference_url

    def embed(self, text: str) -> List[float]:
        res = requests.post(self._endpoint, headers={"Authorization": f"Bearer {self._token}"}, json={"inputs": text})

        if res.status_code != 200:
            raise ValueError(f"Request failed with status code {res.status_code}: {res.text}")
        return res.json()


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()
