import argparse
from pymongo import MongoClient
from modules.pocket import PocketSource


class VectorDatabase():
    def __init__(self, db_name: str):
        self._client = MongoClient()
        self._db = self._client[db_name]

    def fill(self, name: str, documents: dict):
        self._db[name].insert_many(documents)

class Hermes():
    def __init__(self, distance: str):
        self._distance = distance

    def search(self, query):
        # Placeholder function for search functionality
        print(f"Searching for '{query}' using {self._distance} distance...")


def main():
    parser = argparse.ArgumentParser(description="Hermes vector search on local MongoDB")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--distance", "-d", type=str, choices=["knn", "cosine"], default="cosine", help="Distance measure (default: cosine)")
    parser.add_argument("--fill", "-f", action="store_true", help="Fill the database with dummy data")
    args = parser.parse_args()

    #db = VectorDatabase("documents")

    if args.fill:
        with open('data/ril_export.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        source: PocketSource = PocketSource("pocket")
        source_name: str = source.get_name()
        documents: dict = source.get_documents(html_content)
        print(documents)
        #db.fill(source_name, documents)

    hermes = Hermes(args.distance)
    hermes.search(args.query)


if __name__ == "__main__":
    main()
