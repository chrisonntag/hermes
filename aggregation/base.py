from abc import ABC, abstractmethod
from typing import List


class BasePipeline(ABC):
    @abstractmethod
    def build_pipeline(self, query_embedding: List[float]):
        pass

