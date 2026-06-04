"""
BalancedEmbeddings -- Cliente de embeddings via OpenRouter com SmartKeyBalancer.

Usa a API de embeddings compativel do OpenRouter (OpenAI-sdk style) com
balanceamento automatico entre multiplas chaves.

Modelos suportados (via EMBEDDING_MODEL env):
    - nvidia/llama-nemotron-embed-vl-1b-v2:free  (dim: 2048) -- padrao, mais rapido
    - openai/text-embedding-3-small               (dim: 1536)
    - openai/text-embedding-3-large               (dim: 3072)
    - openai/text-embedding-ada-002              (dim: 1536)

Uso:
    from services.utils.balanced_embeddings import setup_embeddings, BalancedEmbeddings

    # Setup unico (carrega chaves do .env e configura o balancer)
    setup_embeddings()

    # Gera embeddings
    embedder = BalancedEmbeddings()
    vectors = embedder.encode(["texto 1", "texto 2"])
"""

import logging
import os
import re
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Registry de dimensoes para modelos conhecidos
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "nvidia/llama-nemotron-embed-vl-1b-v2:free": 2048,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
}

DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


# ---------------------------------------------------------------------------
# Key Loading
# ---------------------------------------------------------------------------

def load_openrouter_keys() -> list[str]:
    """Loads all OPENROUTER_API_KEY* from the environment."""
    keys = [
        v for k, v in os.environ.items()
        if k.startswith("OPENROUTER_API_KEY") and v
    ]
    if not keys:
        raise ValueError(
            "No OPENROUTER_API_KEY found in .env."
        )
    return keys


# ---------------------------------------------------------------------------
# SmartKeyBalancer
# ---------------------------------------------------------------------------

class SmartKeyBalancer:
    """
    Balances requests across multiple OpenRouter keys.

    Comportamento:
    - Round-robin among available keys.
    - Marks key as cooldown when receiving 429 / rate limit.
    - When making a request, automatically falls back through all keys before raising an error.
    """

    def __init__(self, keys: list[str], default_cooldown: int = 60):
        self._keys = list(keys)
        self._index = 0
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._default_cooldown = default_cooldown
        logger.info(f"[EmbeddingsBalancer] {len(keys)} keys loaded.")

    def all_keys(self) -> list[str]:
        """Returns all keys in priority order (available first)."""
        now = time.time()
        available = [k for k in self._keys if self._cooldowns.get(k, 0) <= now]
        cooling = [k for k in self._keys if self._cooldowns.get(k, 0) > now]
        return available + cooling

    def mark_rate_limited(self, key: str, retry_after: int | None = None):
        """Marks key in cooldown due to rate limit (429)."""
        secs = retry_after if retry_after is not None else self._default_cooldown
        with self._lock:
            self._cooldowns[key] = time.time() + secs
        logger.warning(f"[EmbeddingsBalancer] Key ...{key[-6:]} cooldown {secs}s.")

    def mark_failed(self, key: str):
        """Marks key with non-transient error; long cooldown (1h)."""
        with self._lock:
            self._cooldowns[key] = time.time() + 3600
        logger.error(f"[EmbeddingsBalancer] Key ...{key[-6:]} failure (cooldown 1h).")

    def status(self) -> dict:
        """Returns status of all keys (available / cooldown)."""
        now = time.time()
        return {
            f"...{k[-6:]}": (
                "available" if self._cooldowns.get(k, 0) <= now
                else f"cooldown {int(self._cooldowns[k] - now)}s"
            )
            for k in self._keys
        }


# ---------------------------------------------------------------------------
# BalancedEmbeddings
# ---------------------------------------------------------------------------

class BalancedEmbeddings:
    """
    Embeddings client via OpenRouter with automatic balancing.

    - OpenAI-compatible API (POST /v1/embeddings)
    - Automatic key fallback on 429/401
    - Supports batch encoding

    Uso:
        setup_embeddings()
        embedder = BalancedEmbeddings()
        vectors = embedder.encode(["texto 1", "texto 2"])
    """

    def __init__(self, model: str | None = None, dimensions: int | None = None):
        self._balancer = get_balancer()
        self._model = model or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._dimensions = dimensions
        logger.info(f"[BalancedEmbeddings] Model: {self._model}")

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "429" in msg or "rate limit" in msg or "too many requests" in msg

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "401" in msg or "unauthorized" in msg or "invalid api key" in msg

    @staticmethod
    def _parse_retry_after(exc: Exception) -> int | None:
        match = re.search(r"retry.after[\":\s]+(\d+)", str(exc), re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _call_embeddings(self, texts: list[str], key: str) -> list[list[float]]:
        """Calls the OpenRouter embeddings API with a specific key."""
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        kwargs: dict = {"model": self._model, "input": texts}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        response = client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    # -- public API ---------------------------------------------------------

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: List of strings to generate embeddings.

        Returns:
            List of vectors (list[list[float]]).

        Raises:
            RuntimeError: If all keys fail.
        """
        if not texts:
            return []

        keys_to_try = self._balancer.all_keys()
        if not keys_to_try:
            raise RuntimeError("[EmbeddingsBalancer] No key available.")

        last_exc: Exception | None = None

        for key in keys_to_try:
            try:
                logger.debug(f"[EmbeddingsBalancer] Trying key ...{key[-6:]}")
                result = self._call_embeddings(texts, key)
                logger.debug(f"[EmbeddingsBalancer] Success with key ...{key[-6:]}")
                return result
            except Exception as exc:
                last_exc = exc
                if self._is_rate_limit_error(exc):
                    self._balancer.mark_rate_limited(key, self._parse_retry_after(exc))
                    continue
                if self._is_auth_error(exc):
                    self._balancer.mark_failed(key)
                    continue
                raise

        raise RuntimeError(
            f"[EmbeddingsBalancer] All {len(keys_to_try)} keys failed. "
            f"Last error: {last_exc}"
        ) from last_exc

    def encode_single(self, text: str) -> list[float]:
        """Generates embedding for a single text."""
        return self.encode([text])[0]

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def get_embedding_dimension(model: str | None = None) -> int:
        """Returns the expected dimension for known models."""
        m = model or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        return EMBEDDING_DIMENSIONS.get(m, 2048)


# ---------------------------------------------------------------------------
# Module-level globals & setup
# ---------------------------------------------------------------------------

_embeddings_balancer: SmartKeyBalancer | None = None


def get_balancer() -> SmartKeyBalancer:
    """Returns the global balancer, initializing if needed."""
    if _embeddings_balancer is None:
        raise RuntimeError(
            "Embeddings balancer not initialized. "
            "Call setup_embeddings() first."
        )
    return _embeddings_balancer


def setup_embeddings(default_cooldown: int = 60) -> SmartKeyBalancer:
    """
    Loads keys from .env and configures the global embeddings balancer.
    Call at application startup, before instantiating BalancedEmbeddings.
    """
    global _embeddings_balancer
    keys = load_openrouter_keys()
    _embeddings_balancer = SmartKeyBalancer(keys, default_cooldown=default_cooldown)
    return _embeddings_balancer
