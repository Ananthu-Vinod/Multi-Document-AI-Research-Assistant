"""Pick API or embedded local RAG client."""

import os
from typing import Tuple, Union

from components.api_client import RAGApiClient
from components.local_client import LocalRAGClient

ClientType = Union[RAGApiClient, LocalRAGClient]


def _use_local_env() -> bool:
    return os.getenv("USE_LOCAL_RAG", "").strip().lower() in ("1", "true", "yes")


def create_rag_client(
    session_id: str | None = None,
    base_url: str | None = None,
    *,
    prefer_local: bool | None = None,
) -> Tuple[ClientType, str]:
    """
    Return (client, mode) where mode is 'api' or 'local'.

    Uses USE_LOCAL_RAG=true to skip API, otherwise tries API health
  then falls back to embedded local RAG.
    """
    if prefer_local is None:
        prefer_local = _use_local_env()

    if prefer_local:
        client = LocalRAGClient(session_id=session_id)
        return client, "local"

    api = RAGApiClient(base_url=base_url)
    if api.is_reachable():
        return api, "api"

    client = LocalRAGClient(session_id=session_id)
    return client, "local"


def refresh_client_mode(client: ClientType, session_id: str | None = None) -> Tuple[ClientType, str]:
    """Re-check API availability (e.g. after user changes URL)."""
    if isinstance(client, LocalRAGClient) and not _use_local_env():
        api = RAGApiClient()
        if api.is_reachable():
            return api, "api"
        return client, "local"
    if isinstance(client, RAGApiClient):
        if client.is_reachable():
            return client, "api"
        return LocalRAGClient(session_id=session_id), "local"
    return client, "local"
