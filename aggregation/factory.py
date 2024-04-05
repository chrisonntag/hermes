from .base import BaseSimilarityPipeline
from .cosine import CosineSimilarityPipeline
from .dotproduct import DotProductSimilarityPipeline


class SimilarityPipelineFactory:
    @staticmethod
    def get_pipeline(pipeline_type: str, **kwargs) -> BaseSimilarityPipeline:
        if pipeline_type == "cosine":
            return CosineSimilarityPipeline(threshold=0.6, k=4)
        elif pipeline_type == "dotProduct":
            return DotProductSimilarityPipeline(threshold=0.6, k=4, vector_size=1024)
        elif pipeline_type == "euclidean":
            raise NotImplementedError("The euclidean similarity pipeline is not implemented yet.")
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

