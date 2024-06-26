from typing import List
from .base import BaseSimilarityPipeline


class CosineSimilarityPipeline(BaseSimilarityPipeline):
    def __init__(self, threshold: float = 0.5, k: int = 3):
        self._threshold = threshold
        self._k = k

    def build_pipeline(self, embedding_field_name: str, query_embedding: List[float]):
        pipeline = [
            self.add_query_embedding_stage(query_embedding),
            self.calculate_cos_sim_params_stage(embedding_field_name),
            self.calculate_cos_sim_stage(),
            self.filter_threshold_stage("cos_similarity", self._threshold),
            self.sort_stage("cos_similarity"),
            self.limit_stage(self._k)
        ]
        return pipeline 

    def calculate_cos_sim_params_stage(self, embedding_field_name: str):
        """Reduce the query embedding and each stored vector embedding
        to the dot product and the sums of squares of the document and 
        the query embeddings.
        """
        return {
            "$project": {
                "title": 1,
                "cos_similarity_params": {
                    "$reduce": {
                        "input": {"$range": [0, {"$size": f"${embedding_field_name}"}]},
                        "initialValue": {
                            "dot_product": 0,
                            "doc_squared_sum": 0,
                            "query_squared_sum": 0
                        },
                        "in": {
                            "$let": {
                                "vars": {
                                    "doc_embedding": {"$arrayElemAt": [f"${embedding_field_name}", "$$this"]},
                                    "query_embedding": {"$arrayElemAt": ["$query_embedding", "$$this"]}
                                },
                                "in": {
                                    "dot_product": {
                                        "$add": ["$$value.dot_product", {"$multiply": ["$$doc_embedding", "$$query_embedding"]}]
                                    },
                                    "doc_squared_sum": {
                                        "$add": ["$$value.doc_squared_sum", {"$pow": ["$$doc_embedding", 2]}]
                                    },
                                    "query_squared_sum": {
                                        "$add": ["$$value.query_squared_sum", {"$pow": ["$$query_embedding", 2]}]    
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    def calculate_cos_sim_stage(self):
        """Compute cosine similarity by dividing the dot product 
        by the square root of the product of the sums of squares 
        of the document and the query embeddings. 
        """
        return {
            "$project": {
                "title": 1,
                "cos_similarity": {
                    "$divide": [
                        "$cos_similarity_params.dot_product",
                        {
                            "$sqrt": {
                                "$multiply": ["$cos_similarity_params.doc_squared_sum", "$cos_similarity_params.query_squared_sum"]
                            }
                        }
                    ]
                }
            }
        }

