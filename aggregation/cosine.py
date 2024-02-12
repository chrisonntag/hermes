from typing import List
from .base import BasePipeline


class CosineSimilarityPipeline(BasePipeline):
    def __init__(self, threshold: float):
        self._threshold = threshold

    def build_pipeline(self, query_embedding: List[float]):
        pipeline = [
            self.add_query_embedding_stage(query_embedding),
            self.calculate_cos_sim_params_stage(),
            self.calculate_cos_sim_stage(),
            self.filter_cos_sim_threshold_stage(self._threshold)
        ]
        return pipeline

    def add_query_embedding_stage(self, query_embedding: List[float]):
        """Add the embedding of the query as a field to each document 
        in the collection.
        """
        return {
            "$addFields": {
                "query_embedding": query_embedding
            }
        }

    def filter_cos_sim_threshold_stage(self, threshold: float):
        """Matches the field cos_sim to include all documents 
        with similarity greater than a given threshold. 
        """
        return {
            "$match": {
                "cos_similarity": {"$gt": threshold}
            }
        }

    def calculate_cos_sim_params_stage(self):
        """Reduce the query embedding and each stored vector embedding
        to the dot product and the sums of squares of the document and 
        the query embeddings.
        """
        return {
            "$project": {
                "title": 1,
                "cos_similarity_params": {
                    "$reduce": {
                        "input": {"$range": [0, {"$size": "$vectorEmbedding"}]},
                        "initialValue": {
                            "dot_product": 0,
                            "doc_squared_sum": 0,
                            "query_squared_sum": 0
                        },
                        "in": {
                            "$let": {
                                "vars": {
                                    "doc_embedding": {"$arrayElemAt": ["$vectorEmbedding", "$$this"]},
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

