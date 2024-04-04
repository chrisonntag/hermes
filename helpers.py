from typing import Any, Dict, List, Optional
from modules.pocket import Pocket
from embedding.base import HuggingFaceEmbedder
from datasets import load_dataset


def fill_database_pocket(db, collection_name: str = "pocket") -> None:
    """
    Fill the database with embedded documents from Pocket.

    Args:
        db (MongoDBVectorStore): MongoDB connection
    """
    print("Load documents")
    with open('data/ril_export.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    source: DataSource = Pocket(collection_name)
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
    db.delete_many(collection_name, {})
    db.add_many(collection_name, documents)


def fill_database_embedded_movies(db, collection_name: str = "movies") -> None:
    """
    Fill the database with embedded movies from HuggingFace dataset
    christophsonntag/gte_embedded_movies

    Args:
        db (MongoDBVectorStore): MongoDB connection
    """
    dataset = load_dataset("christophsonntag/gte_embedded_movies")
    documents: list[Dict] = dataset["train"]

    print("Add documents...")
    db.delete_many(collection_name, {})
    db.add_many(collection_name, documents)
