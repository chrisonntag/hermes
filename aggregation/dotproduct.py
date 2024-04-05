from .base import BaseSimilarityPipeline
from .cosine import CosineSimilarityPipeline
from typing import List


class DotProductSimilarityPipeline(CosineSimilarityPipeline):
    def __init__(self, threshold: float = 0.5, k: int = 3, vector_size: int = 1024):
        self._threshold = threshold
        self._k = k
        self._vector_size = vector_size

    def build_pipeline(self, embedding_field_name: str, query_embedding: List[float]):
        pipeline = [
            self.add_query_embedding_stage(query_embedding),
            self.calculate_cos_sim_params_stage(embedding_field_name),
            self.calculate_cos_sim_stage(),
            self.dot_product_vector_length_stage(),
            self.filter_threshold_stage("dot_product_similarity", self._threshold),
            self.sort_stage("dot_product_similarity"),
            self.limit_stage(self._k)
        ]
        return pipeline

    def dot_product_vector_length_stage(self):
        """
        This returns the aggregation stage for calculating the dot product 
        by multiplying the existing field cos_similarity from the cosine 
        distance with the length of the query_embedding and the length of 
        the doc_embedding. 
        """
        return {
            "$project": {
                "title": 1,
                "dot_product_similarity": {
                    "$multiply": [
                        "$cos_similarity",
                        self._vector_size*self._vector_size,
                    ]
                }
            }
        }
       
