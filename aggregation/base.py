from abc import ABC, abstractmethod
from typing import List


class BaseSimilarityPipeline(ABC):
    @staticmethod
    def add_query_embedding_stage(query_embedding: List[float]) -> dict:
        """Add the embedding of the query as a field to each document 
        in the collection.

        Args:
            query_embedding (List[float]): The vector embedding of the 
            query used to find similar documents.
        """
        return {
            "$addFields": {
                "query_embedding": query_embedding
            }
        }
       
    @staticmethod
    def filter_threshold_stage(field_name: str, threshold: float):
        """Matches the field_name to include all documents 
        with similarity greater than a given threshold. 
        """
        return {
            "$match": {
                field_name: {"$gt": threshold}
            }
        }

    @staticmethod
    def sort_stage(sort_field: str, order: int = -1) -> dict:
        """
        Sort documents by a given field in (default: descending) order.

        Args:
            sort_field (str): The field to sort by.
        """
        return {"$sort": {sort_field: order}}

    @staticmethod
    def limit_stage(limit: int) -> dict:
        """
        Limit the number of documents returned.

        Args:
            limit (int): The number of documents to return.
        """
        return {"$limit": limit}

    @abstractmethod
    def build_pipeline(self, query_embedding: List[float]):
        pass


