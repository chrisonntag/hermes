import argparse
from pymongo import MongoClient
from typing import Any, Dict, List, Optional
from modules.pocket import PocketSource


class MongoDBVectorStore():
    """Vector Store on top of MongoDB
    """
    def __init__(self, 
                 mongodb_client: Optional[Any] = None, 
                 db_name: str = "documents"):
        if mongodb_client is not None:
            self._client = mongodb_client
        else:
            self._client: Dict = {}  # Change this to MongoClient()

        self._db_name = db_name
        self._client[db_name] = []  # Remove this on move to MongoClient
        self._db: List[Dict] = self._client[db_name]

    def get(self, doc_ind: int) -> Dict:
        """Get embedding."""
        return self._db[doc_ind]

    def add(self, docs: List[Dict]) -> None:
        """Add docs to index."""
        for doc in docs:
            self._db.append(doc)

    def delete(self, doc_ind: str) -> None:
        """ Delete nodes using with doc_ind."""
        pass

    def query(self, embedding: List[float]) -> List[Dict]:
        """Get docs for response.
        
        Returns:
            Dummy data
        """
        return self._db[0:3]

    def persist(self, persist_path, fs=None) -> None:
        """Persist the VectorStore."""
        pass

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
        results: List[Dict] = self._store.query(embedding)

        return results


def main():
    parser = argparse.ArgumentParser(description="Hermes vector search on local MongoDB")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--distance", "-d", type=str, choices=["knn", "cosine"], default="cosine", help="Distance measure (default: cosine)")
    parser.add_argument("--fill", "-f", action="store_true", help="Fill the database with dummy data")
    args = parser.parse_args()

    db = MongoDBVectorStore(db_name = "documents")

    if args.fill:
        print("Load documents")
        with open('data/ril_export.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        source: PocketSource = PocketSource("pocket")
        source_name: str = source.get_name()
        documents: dict = source.get_documents(html_content)

        print("Create embeddings")
        

        print("Add documents...")
        db.add(documents)

    hermes = Hermes(vector_store=db, distance=args.distance)
    results = hermes.search(args.query)

    print(results)


if __name__ == "__main__":
    main()
