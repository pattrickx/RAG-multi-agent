"""
PDF scraper — PDF processing with semantic chunking.

Flow:
    1. PDF -> pages (pypdf)
    2. Page -> structured text (docling)
    3. Text -> semantic chunks (embeddings via OpenRouter)
    4. Chunks -> Qdrant (via qdrant_client)
"""

import hashlib
import os
import tempfile
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from src.files_loders.keywords import LargeCorpusKeywordExtractor
from src.files_loders.qdrant_client import send_chunks_to_qdrant
from services.utils.balanced_embeddings import BalancedEmbeddings, setup_embeddings

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

API_BASE = os.environ.get("API_BASE_URL")
API_USER = os.environ.get("API_USER")
API_PASS = os.environ.get("API_PASS")

# Docling pipeline config
_PIPELINE_OPTIONS = PdfPipelineOptions()
_PIPELINE_OPTIONS.do_code_enrichment = True
_PIPELINE_OPTIONS.do_formula_enrichment = True

# Chunking defaults
DEFAULT_MAX_CHUNK_SIZE = 1000
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_MIN_CHUNK_SIZE = 200


# ---------------------------------------------------------------------------
# Embeddings (lazy init)
# ---------------------------------------------------------------------------

_embedder: BalancedEmbeddings | None = None


def _get_embedder() -> BalancedEmbeddings:
    """Returns the embedder, initializing if needed (lazy)."""
    global _embedder
    if _embedder is None:
        setup_embeddings()
        _embedder = BalancedEmbeddings(model=EMBEDDING_MODEL)
    return _embedder


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def generate_file_hash(source: str) -> str:
    """Generates MD5 hash of the file (local or URL)."""
    if source.startswith("http://") or source.startswith("https://"):
        import requests
        response = requests.get(source)
        file_content = response.content
    else:
        with open(source, "rb") as f:
            file_content = f.read()
    return hashlib.md5(file_content).hexdigest()


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def _split_sentences(text: str) -> list[str]:
    """Splits text into sentences (nltk with fallback)."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.split(".") if s.strip()]


def _chunk_by_size(text: str, max_size: int) -> list[str]:
    """Splits text into chunks by size (fallback without embedding)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start and (end - last_space) < 100:
                end = last_space
        chunks.append(text[start:end])
        start = end
    return chunks


def semantic_chunking_with_limits(
    chunks: list[dict[str, Any]],
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """
    Applies semantic chunking with character limit.
    Uses embeddings via OpenRouter for similarity calculation.

    Args:
        chunks: List of dicts with 'text', 'page', 'type', 'level', 'provenance'.
        max_chunk_size: Maximum chunk size in characters.
        similarity_threshold: Similarity threshold (0-1). Below = new chunk.
        min_chunk_size: Minimum chunk size to apply semantic chunking.

    Returns:
        List of processed chunks with preserved metadata.
    """
    if not chunks:
        return []

    embedder = _get_embedder()

    # Group by page
    page_groups: dict[int, list[dict]] = {}
    for chunk in chunks:
        page = chunk.get("page", 0)
        page_groups.setdefault(page, []).append(chunk)

    processed_chunks: list[dict[str, Any]] = []

    for page, page_chunks in page_groups.items():
        page_text = " ".join(c["text"] for c in page_chunks)

        # Coletar provenance
        provenance_list = []
        for chunk in page_chunks:
            prov = chunk.get("provenance")
            if prov:
                if isinstance(prov, list):
                    provenance_list.extend(prov)
                else:
                    provenance_list.append(prov)

        # Too short text: keep as-is
        if len(page_text) < min_chunk_size:
            processed_chunks.append({
                "text": page_text,
                "page": page,
                "type": "combined_page",
                "level": None,
                "provenance": provenance_list,
                "chunk_strategy": "direct",
            })
            continue

        sentences = _split_sentences(page_text)

        if len(sentences) <= 1:
            processed_chunks.append({
                "text": page_text,
                "page": page,
                "type": "combined_page",
                "level": None,
                "provenance": provenance_list,
                "chunk_strategy": "direct",
            })
            continue

        # Generate embeddings for sentences
        try:
            sentence_embeddings = embedder.encode(sentences)
        except Exception as e:
            print(f"Embedding error: {e}. Using size fallback.")
            for chunk_text in _chunk_by_size(page_text, max_chunk_size):
                processed_chunks.append({
                    "text": chunk_text,
                    "page": page,
                    "type": "combined_page",
                    "level": None,
                    "provenance": provenance_list,
                    "chunk_strategy": "size_fallback",
                })
            continue

        # Chunking semantico: agrupa sentencas similares
        semantic_chunks: list[str] = []
        current_chunk: list[str] = []
        current_text = ""

        for i, sentence in enumerate(sentences):
            if i > 0:
                similarity = _cosine_similarity(
                    sentence_embeddings[i - 1], sentence_embeddings[i]
                )
                new_size = len(current_text) + len(sentence) + 1

                if (similarity < similarity_threshold or
                        (new_size > max_chunk_size and current_chunk)):
                    if current_chunk:
                        semantic_chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_text = sentence
                else:
                    current_chunk.append(sentence)
                    current_text += " " + sentence if current_text else sentence
            else:
                current_chunk = [sentence]
                current_text = sentence

        if current_chunk:
            semantic_chunks.append(" ".join(current_chunk))

        # Post-processing: split chunks exceeding max_chunk_size
        final_chunks: list[str] = []
        for chunk_text in semantic_chunks:
            if len(chunk_text) <= max_chunk_size:
                final_chunks.append(chunk_text)
            else:
                final_chunks.extend(_chunk_by_size(chunk_text, max_chunk_size))

        for chunk_text in final_chunks:
            processed_chunks.append({
                "text": chunk_text,
                "page": page,
                "type": "semantic_chunk",
                "level": None,
                "provenance": provenance_list,
                "chunk_strategy": "semantic",
                "similarity_threshold": similarity_threshold,
                "max_chunk_size": max_chunk_size,
            })

    return processed_chunks


def process_docling_chunks(
    chunks: list[dict[str, Any]], **kwargs: Any
) -> list[dict[str, Any]]:
    """Wrapper: processes Docling chunks with semantic chunking."""
    return semantic_chunking_with_limits(chunks, **kwargs)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _set_file_status(
    client: httpx.Client,
    file_id: int,
    status: str,
    loaded_pages: int = -1,
    total_pages: int = -1,
) -> None:
    """Updates file status via API."""
    params: dict[str, Any] = {"file_status": status}
    if loaded_pages >= 0:
        params["loaded_pages"] = loaded_pages
    if total_pages >= 0:
        params["total_pages"] = total_pages

    resp = client.put(f"{API_BASE}/files/{file_id}/status/", params=params)
    resp.raise_for_status()
    print(
        f"File ID {file_id} status updated to '{status}' "
        f"(loaded_pages={loaded_pages}, total_pages={total_pages})"
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pdf(
    source: str,
    collection_name: str,
    file_id: int,
    source_origin_name: str | None = None,
    start_id: int = 0,
    start_page: int = 0,
) -> int:
    """
    Processes a PDF page by page: extracts text, generates semantic chunks
    and sends to Qdrant.

    Args:
        source: Caminho do PDF.
        collection_name: Qdrant collection name.
        file_id: File ID in the API.
        source_origin_name: Display name (default: basename of source).
        start_id: Initial ID for points in Qdrant.
        start_page: Initial page (1-indexed).

    Returns:
        Last ID used in Qdrant, or -1 on error.
    """
    if source_origin_name is None:
        source_origin_name = os.path.basename(source)

    file_hash = generate_file_hash(source)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_PIPELINE_OPTIONS)
        }
    )
    embedder = _get_embedder()

    try:
        reader = PdfReader(source)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return -1

    print(f"Processing PDF: {source_origin_name} ({len(reader.pages)} pages)")

    last_id = start_id

    with httpx.Client(auth=(API_USER, API_PASS), timeout=30.0) as client:
        _set_file_status(client, file_id, "in_progress", total_pages=len(reader.pages))

        for page_idx, page in tqdm(
            enumerate(reader.pages[start_page:], start=start_page + 1),
            total=len(reader.pages[start_page:]),
            desc="Processing pages",
        ):
            tmp_path = None
            try:
                # Extract page to temporary PDF
                with tempfile.NamedTemporaryFile(
                    suffix=f"_page_{page_idx}.pdf", delete=False
                ) as tmp_file:
                    writer = PdfWriter()
                    writer.add_page(page)
                    writer.write(tmp_file)
                    tmp_path = tmp_file.name

                # Convert with docling
                page_result = converter.convert(tmp_path)
                doc = page_result.document

                # Extract text and keywords
                content = [
                    getattr(item, "text", "")
                    for item, _ in doc.iterate_items()
                    if getattr(item, "text", None)
                ]
                keyword_extractor = LargeCorpusKeywordExtractor(window_size=5, n_jobs=-1)
                keyword_extractor.fit(content)
                keywords = keyword_extractor.extract_keywords(top_k=15)

                # Extract chunks from page
                page_chunks: list[dict[str, Any]] = []
                for item, level in doc.iterate_items():
                    text = getattr(item, "text", None)
                    if not text or not text.strip():
                        continue
                    page_chunks.append({
                        "text": text.strip(),
                        "page": page_idx,
                        "type": type(item).__name__,
                        "level": level,
                        "provenance": getattr(item, "prov", None),
                    })

                if not page_chunks:
                    continue

                # Chunking semantico
                semantic_chunks = semantic_chunking_with_limits(
                    page_chunks,
                    max_chunk_size=DEFAULT_MAX_CHUNK_SIZE,
                    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                    min_chunk_size=DEFAULT_MIN_CHUNK_SIZE,
                )
                if not semantic_chunks:
                    continue

                # Generate embeddings and send to Qdrant
                texts = [c["text"] for c in semantic_chunks if c.get("text")]
                if not texts:
                    continue

                embeddings = embedder.encode(texts)
                last_id = send_chunks_to_qdrant(
                    semantic_chunks, embeddings, source_origin_name,
                    file_hash, collection_name=collection_name,
                    keywords=keywords, start_id=start_id,
                )
                start_id = last_id + 1
                print(f"Page {page_idx} sent. Last ID: {last_id}")
                _set_file_status(client, file_id, "in_progress", loaded_pages=page_idx)

            except Exception as e:
                print(f"Error on page {page_idx}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    return last_id
