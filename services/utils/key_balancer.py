"""
SmartKeyBalancer -- Balanceamento de chaves OpenRouter para LLM chat.

Fornece `BalancedOpenRouter` (ChatOpenAI com fallback entre chaves) e
`SmartKeyBalancer` (round-robin + cooldown).

Modelos LLM suportados (via LLM_MODEL env):
    - nvidia/nemotron-3-ultra-550b-a55b:free  (550B MoE, reasoning) -- padrao
    - nvidia/nemotron-3-super-120b-a12b:free   (120B MoE, rapido)
    - openai/gpt-4o
    - anthropic/claude-3-5-sonnet

Uso:
    from services.utils.key_balancer import setup_balancer, create_llm

    setup_balancer()
    llm = create_llm()
    response = llm.invoke("Ola, tudo bem?")
"""

import logging
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

load_dotenv()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


# ---------------------------------------------------------------------------
# Key Loading
# ---------------------------------------------------------------------------

def load_openrouter_keys() -> list[str]:
    """Carrega todas as OPENROUTER_API_KEY* do ambiente e valida formato."""
    keys = [
        v for k, v in os.environ.items()
        if k.startswith("OPENROUTER_API_KEY") and v
    ]

    if not keys:
        raise ValueError(
            "No OPENROUTER_API_KEY found in .env.\n"
            "Exemplo: OPENROUTER_API_KEY, OPENROUTER_API_KEY_2, ..."
        )

    invalid = [k for k in keys if not k.startswith("sk-or-")]
    if invalid:
        raise ValueError(
            f"{len(invalid)} chave(s) com formato invalido (esperado: sk-or-...)."
        )

    logger.info(f"[KeyBalancer] {len(keys)} keys loaded.")
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
    - Faz fallback automatico por todas as chaves antes de lancar erro.
    """

    def __init__(self, keys: list[str], default_cooldown: int = 60):
        self._keys = list(keys)
        self._index = 0
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._default_cooldown = default_cooldown

    def next_key(self) -> str | None:
        """Retorna a proxima chave available (fora de cooldown), ou None."""
        with self._lock:
            now = time.time()
            for _ in range(len(self._keys)):
                key = self._keys[self._index % len(self._keys)]
                self._index += 1
                if self._cooldowns.get(key, 0) <= now:
                    return key
            return None

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
        logger.warning(f"[KeyBalancer] Key ...{key[-6:]} cooldown {secs}s.")

    def mark_failed(self, key: str):
        """Marks key with non-transient error; long cooldown (1h)."""
        with self._lock:
            self._cooldowns[key] = time.time() + 3600
        logger.error(f"[KeyBalancer] Key ...{key[-6:]} failure (cooldown 1h).")

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
# Error helpers
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "401" in msg or "unauthorized" in msg or "invalid api key" in msg


def _parse_retry_after(exc: Exception) -> int | None:
    """Tries to extract the Retry-After value from the error body."""
    match = re.search(r"retry.after[\":\s]+(\d+)", str(exc), re.IGNORECASE)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# BalancedOpenRouter
# ---------------------------------------------------------------------------

class BalancedOpenRouter(ChatOpenAI):
    """
    ChatOpenAI com balanceamento automatico entre multiplas chaves OpenRouter.

    - Tenta todas as chaves disponiveis antes de lancar erro.
    - 429 -> marca cooldown e tenta proxima chave imediatamente.
    - 401 -> marca falha longa (1h) e tenta proxima chave.
    - Qualquer outro erro -> propaga imediatamente.
    """

    _balancer: SmartKeyBalancer | None = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs,
    ) -> ChatResult:
        if self._balancer is None:
            raise RuntimeError("BalancedOpenRouter._balancer nao foi configurado.")

        keys_to_try = self._balancer.all_keys()
        if not keys_to_try:
            raise RuntimeError("[KeyBalancer] No key available.")

        last_exc: Exception | None = None

        for key in keys_to_try:
            self.openai_api_key = key
            logger.debug(f"[KeyBalancer] Trying key ...{key[-6:]}")
            try:
                result = super()._generate(messages, stop, run_manager, **kwargs)
                logger.debug(f"[KeyBalancer] Success with key ...{key[-6:]}")
                return result
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc):
                    self._balancer.mark_rate_limited(key, _parse_retry_after(exc))
                    continue
                if _is_auth_error(exc):
                    self._balancer.mark_failed(key)
                    continue
                raise

        raise RuntimeError(
            f"[KeyBalancer] All {len(keys_to_try)} keys failed. "
            f"Last error: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_llm(model: str | None = None, **kwargs) -> BalancedOpenRouter:
    """
    Retorna um ChatModel pronto para uso com balanceamento de chaves.

    Args:
        model: Nome do modelo OpenRouter. Se None, le do env LLM_MODEL.
        **kwargs: kwargs extras para ChatOpenAI (temperature, etc).

    Returns:
        BalancedOpenRouter configurado.

    Uso:
        llm = create_llm()
        llm = create_llm("nvidia/nemotron-3-ultra-550b-a55b:free")
        llm = create_llm("openai/gpt-4o", temperature=0)
    """
    resolved_model = model or os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
    return BalancedOpenRouter(
        base_url="https://openrouter.ai/api/v1",
        api_key="placeholder",  # sobrescrito a cada chamada
        model=resolved_model,
        **kwargs,
    )


def setup_balancer(default_cooldown: int = 60) -> SmartKeyBalancer:
    """
    Carrega as chaves do .env e configura o balancer global.
    Chame no inicio da aplicacao, antes de instanciar BalancedOpenRouter.
    """
    keys = load_openrouter_keys()
    balancer = SmartKeyBalancer(keys, default_cooldown=default_cooldown)
    BalancedOpenRouter._balancer = balancer
    return balancer


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    balancer = setup_balancer()
    print("Status inicial:", balancer.status())

    llm = create_llm()

    # Uso direto:
    # response = llm.invoke("Ola, tudo bem?")
    # print(response.content)

    # Uso no LangGraph:
    # from langgraph.graph import StateGraph, MessagesState
    #
    # def agent_node(state: MessagesState):
    #     return {"messages": [llm.invoke(state["messages"])]}
    #
    # graph = StateGraph(MessagesState)
    # graph.add_node("agent", agent_node)
    # graph.set_entry_point("agent")
    # graph.set_finish_point("agent")
    # app = graph.compile()
    #
    # result = app.invoke({"messages": [("user", "Qual e a capital do Brasil?")]})
    # print(result["messages"][-1].content)
