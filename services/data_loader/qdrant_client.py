"""
Qdrant client — NER local + embeddings via OpenRouter.

Funcoes:
    create_collection()       — creates Qdrant collection
    send_chunks_to_qdrant()   — inserts chunks with embeddings
    get_chunks_from_qdrant() — similarity search
"""

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from services.utils.balanced_embeddings import BalancedEmbeddings, setup_embeddings

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

DEFAULT_COLLECTION = "docs"
DEFAULT_GROUPS = ["master"]

NER_MODEL_NAME = "Davlan/bert-base-multilingual-cased-ner-hrl"


# ---------------------------------------------------------------------------
# Clients (module-level singletons)
# ---------------------------------------------------------------------------

# NER pipeline (modelo local leve)
_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
_ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
_nlp = pipeline("ner", model=_ner_model, tokenizer=_tokenizer)

# Qdrant client
_client = QdrantClient(url=QDRANT_URL)

# Embeddings via OpenRouter (lazy init)
_embedder: BalancedEmbeddings | None = None


def _get_embedder() -> BalancedEmbeddings:
    """Returns the embedder, initializing if needed (lazy)."""
    global _embedder
    if _embedder is None:
        setup_embeddings()
        _embedder = BalancedEmbeddings(model=EMBEDDING_MODEL)
    return _embedder


# ---------------------------------------------------------------------------
# NER helpers
# ---------------------------------------------------------------------------

def extract_words_and_types(text: str) -> list[str]:
    """
    Extracts named entities from text using local NER.
    Returns list of unique words with type (PER, LOC, ORG, MISC).
    """
    entities = _nlp(text)
    words_and_types: list[dict] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    for entity in entities:
        tag = entity.get("entity", "")
        if "-" not in tag:
            continue

        prefix, ent_type = tag.split("-", 1)

        if prefix == "B":
            if current_type is not None:
                words_and_types.append({
                    "type": current_type,
                    "word": text[current_start:current_end],
                    "start": current_start,
                    "end": current_end,
                })
            current_type = ent_type
            current_start = entity["start"]
            current_end = entity["end"]

        elif prefix == "I" and current_type == ent_type and current_start is not None:
            current_end = entity["end"]

        else:
            if current_type is not None:
                words_and_types.append({
                    "type": current_type,
                    "word": text[current_start:current_end],
                    "start": current_start,
                    "end": current_end,
                })
            current_type = None
            current_start = None
            current_end = None

    if current_type is not None:
        words_and_types.append({
            "type": current_type,
            "word": text[current_start:current_end].lower(),
            "start": current_start,
            "end": current_end,
        })

    words = {
        item["word"].lower().replace("\n", " ").replace("\r", " ")
        for item in words_and_types
    }
    return list(words)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_collection(
    name_collection: str,
    size_vector: int | None = None,
    distance: Distance = Distance.COSINE,
) -> None:
    """
    Creates a Qdrant collection. If size_vector is not provided,
    auto-detects from the configured embedding model.
    """
    try:
        existing = _client.get_collections().collections
        if any(c.name == name_collection for c in existing):
            print(f"Collection '{name_collection}' already exists.")
            return

        if size_vector is None:
            size_vector = BalancedEmbeddings.get_embedding_dimension(EMBEDDING_MODEL)

        _client.create_collection(
            collection_name=name_collection,
            vectors_config=VectorParams(size=size_vector, distance=distance),
        )
        print(f"Collection '{name_collection}' created (dim={size_vector}).")
    except Exception as e:
        print(f"Error creating collection '{name_collection}': {e}")


def send_chunks_to_qdrant(
    chunks_semanticos: list[dict],
    embeddings_list: list[list[float]],
    source: str,
    file_hash: str,
    keywords: list | None = None,
    start_id: int = 0,
    collection_name: str = DEFAULT_COLLECTION,
    groups: list | None = None,
) -> int:
    """
    Sends chunks with embeddings to Qdrant.

    Args:
        chunks_semanticos: List of dicts with 'text', 'page', etc.
        embeddings_list: Corresponding vector list (list[float]).
        source: Source file name.
        file_hash: Unique file hash.
        keywords: List of (keyword, score) for extra tags.
        start_id: Initial ID for points.
        collection_name: Qdrant collection name.
        groups: Access groups (default: ["master"]).

    Returns:
        Last ID used.
    """
    if groups is None:
        groups = DEFAULT_GROUPS

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks_semanticos, embeddings_list)):
        tags = extract_words_and_types(chunk["text"])
        text_lower = chunk["text"].lower().replace("\n", " ").replace("\r", " ")

        if keywords:
            for kw, _ in keywords:
                if kw in text_lower and kw not in tags:
                    tags.append(kw)

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "page": chunk["page"],
                "id": start_id + i,
                "in_page_id": i,
                "tags": tags,
                "file_hash": file_hash,
                "source": source,
                "group": collection_name,
                "created_at": datetime.now().isoformat(),
                "word_count": len(chunk["text"].split()),
                "char_count": len(chunk["text"]),
            },
        ))

    _client.upsert(collection_name=collection_name, wait=True, points=points)
    print(f"Inserted {len(points)} chunks from {source}")
    return start_id + len(points) - 1


def get_chunks_from_qdrant(
    query_text: str,
    collection_name: str = DEFAULT_COLLECTION,
    groups: list | None = None,
    top_k: int = 10,
    key_words: str | None = None,
    file_hash: str | None = None,
):
    """
    Searches chunks in Qdrant using embeddings via OpenRouter.

    Args:
        query_text: Query text.
        collection_name: Collection name.
        groups: Group filter.
        top_k: Maximum number of results.
        key_words: Keywords for should filter.
        file_hash: File hash filter.

    Returns:
        Qdrant search result.
    """
    if groups is None:
        groups = DEFAULT_GROUPS

    embedder = _get_embedder()
    query_embedding = embedder.encode_single(query_text)

    must_filters = [{"key": "groups", "match": {"value": groups}}]
    if file_hash:
        must_filters.append({"key": "file_hash", "match": {"value": file_hash}})

    should_filters = []
    if key_words:
        should_filters.append({"key": "text", "match": {"value": key_words}})

    return _client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        filter={"must": must_filters, "should": should_filters},
    )
