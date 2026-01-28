# src/genai/retrivel/pince_ret.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env file")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def query_pinecone(query: str, namespace: str, top_k: int = 3):
    if not namespace:
        raise ValueError("Namespace cannot be empty.")
    query_vector = embed_model.encode(query).tolist()
    try:
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        matches = results.matches
        if not matches:
            return []
        formatted = []
        for match in matches:
            meta = match.metadata or {}
            page_number = meta.get("page_number", 0)
            try:
                page_label = int(page_number) + 1
            except ValueError:
                page_label = 0
            formatted.append({
                "id": match.id,
                "score": match.score,
                "namespace": namespace,
                "source": meta.get("pdf_name", meta.get("source_file", "Unknown")),
                "text": meta.get("text", meta.get("page_content", "")),
                "page": page_label
            })
        return formatted
    except Exception as e:
        print(f"Pinecone query failed for namespace '{namespace}': {e}")
        return []


def get_all_namespaces():
    try:
        stats = index.describe_index_stats()
        if not stats or not stats.get("namespaces"):
            return []
        return list(stats["namespaces"].keys())
    except Exception as e:
        print(f"Could not fetch namespaces: {e}")
        return []
