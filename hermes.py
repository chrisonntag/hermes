import argparse
import pymongo
from typing import Any, Dict, List, Optional
from modules.pocket import Pocket
from embedding.base import HuggingFaceEmbedder
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

    def add(self, collection: str, docs: List[Dict]) -> pymongo.results.InsertManyResult:
        """Add docs to index."""
        return self._db[collection].insert_many(docs)

    def delete(self, collection: str, query: dict) -> pymongo.results.DeleteResult:
        """ Delete nodes using with doc_ind."""
        return self._db[collection].delete_one(query)

    def query(self, collection: str, query: dict) -> List[Dict]:
        """Get docs for response.
        """
        return list(self._db[collection].find(query))

    def aggregate(self, pipeline: List[Dict]) -> List[Dict]:
        return list(self._db[collection].aggregate(pipeline))


class Hermes():
    def __init__(self, 
                 vector_store: MongoDBVectorStore,
                 distance: str):
        self._distance = distance
        self._store = vector_store

    def search(self, query: str) -> List[Dict]:
        """Search for a given query

        This creates an embedding from the query and checks the VectorStore for semantically similar documents. 

        Args:
            query: str
        """
        # Create the embedding
        embedding: List[float] = [0.0, 0.1]

        print(f"Searching for '{query}' using {self._distance} distance...")
        query_dict: dict = {'title': query}
        results: List[Dict] = self._store.query("pocket", query_dict)

        return results

def fill_database(db: MongoDBVectorStore):
    print("Load documents")
    with open('data/ril_export.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    source: DataSource = Pocket("pocket")
    documents: List[Dict] = source.extract_documents(html_content, stop_after=3, follow_links=True)

    print("Create embeddings")
    hf = HuggingFaceEmbedder(
            access_token=HERMES_CONFIG["hf_access_token"], 
            inference_url=HERMES_CONFIG["hf_inference_endpoint"]) 
    for doc in documents:
        # Choose which information should be embedded.
        # Load whole HTMl page of link. 
        doc["vectorEmbedding"] = hf.generate(doc["content"])

    print("Add documents...")
    db.add("pocket", documents)

def main():
    parser = argparse.ArgumentParser(description="Hermes vector search on local MongoDB")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--distance", "-d", type=str, choices=["knn", "cosine"], default="cosine", help="Distance measure (default: cosine)")
    parser.add_argument("--fill", "-f", action="store_true", help="Fill the database with dummy data")
    args = parser.parse_args()

    db = MongoDBVectorStore(db_name = "documents")

    if args.fill:
        fill_database(db)
        
    hermes = Hermes(vector_store=db, distance=args.distance)
    results = hermes.search(args.query)

    print(results)


if __name__ == "__main__":
    main()
