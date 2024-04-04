import argparse
import time
import pymongo
from typing import Any, Dict, List, Optional
from modules.pocket import Pocket
from embedding.base import HuggingFaceEmbedder, BaseEmbedder, SentenceTransformerEmbedder
from helpers import fill_database_embedded_movies
from aggregation.cosine import CosineSimilarityPipeline
from config import HERMES_CONFIG


class MongoDBVectorStore():
    """Vector Store on top of MongoDB
    """
    def __init__(self, 
                 mongodb_client: Optional[Any] = None, 
                 db_name: str = "documents"):
        if mongodb_client is not None:
            self._client = mongodb_client
        else:
            mongo_uri: str = "mongodb://%s:%d/" % (HERMES_CONFIG['mongodb']['host'], HERMES_CONFIG['mongodb']['port'])
            self._client: pymongo.MongoClient = pymongo.MongoClient(mongo_uri)

        self._db_name = db_name
        self._db = self._client[db_name]

    def get(self, collection: str, query: dict) -> Optional[Dict]:
        """Get embedding."""
        return self._db[collection].find_one(query)

    def add_many(self, collection: str, docs: List[Dict]) -> pymongo.results.InsertManyResult:
        """Add docs to index."""
        return self._db[collection].insert_many(docs)

    def delete(self, collection: str, query: dict) -> pymongo.results.DeleteResult:
        """ Delete nodes using with doc_ind."""
        return self._db[collection].delete_one(query)

    def delete_many(self, collection: str, query: dict) -> pymongo.results.DeleteResult:
        """ Delete nodes using with doc_ind."""
        return self._db[collection].delete_many(query)

    def query(self, collection: str, query: dict) -> List[Dict]:
        """Get docs for response.
        """
        return list(self._db[collection].find(query))

    def aggregate(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        return list(self._db[collection].aggregate(pipeline))


class Hermes():
    def __init__(self, 
                 vector_store: MongoDBVectorStore,
                 collection_name: str,
                 embedding_field_name: str,
                 distance: str, 
                 embedder: BaseEmbedder):
        self._distance = distance
        self._store = vector_store
        self._embedder = embedder
        self._collection_name = collection_name
        self._embedding_field_name = embedding_field_name

    def search(self, query: str) -> List[Dict]:
        """Search for a given query

        This creates an embedding from the query and checks the VectorStore for semantically similar documents. 

        Args:
            query: str
        """
        print(f"Searching for '{query}' using {self._distance} distance...")
        # Create the embedding 
        query_embedding: List[float] = self._embedder.embed(query)

        query_pipeline = CosineSimilarityPipeline(threshold=0.6, k=4).build_pipeline(self._embedding_field_name, query_embedding)
        results: List[Dict] = self._store.aggregate(self._collection_name, query_pipeline)

        return results


def main():
    parser = argparse.ArgumentParser(description="Hermes vector search on local MongoDB")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--distance", "-d", type=str, choices=["euclidean", "cosine", "dotProduct"], default="cosine", help="Similarity measure (default: cosine)")
    parser.add_argument("--fill", "-f", action="store_true", help="Fill the database with dummy data")
    args = parser.parse_args()

    db = MongoDBVectorStore(db_name = "documents")

    if args.fill:
        fill_database_embedded_movies(db, "movies")
        
    hermes = Hermes(vector_store=db, collection_name="movies", embedding_field_name="plot_embedding", distance=args.distance, embedder=SentenceTransformerEmbedder(HERMES_CONFIG['embedder_identifier']))

    start_time = time.time()

    results = hermes.search(args.query)

    end_time = time.time()
    elapsed_time = end_time - start_time

    for result in results:
        print(result)

    print("Elapsed time:", elapsed_time, "seconds")


if __name__ == "__main__":
    main()
