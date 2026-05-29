"""HTTP client for the FastAPI RAG backend."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "300"))
HEALTH_TIMEOUT = int(os.getenv("API_HEALTH_TIMEOUT_SECONDS", "3"))


class APIError(Exception):
    """Raised when the backend returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RAGApiClient:
    """Thin wrapper around REST endpoints."""

    mode = "api"

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or DEFAULT_BACKEND).rstrip("/")
        self.last_stream_meta: Dict[str, Any] = {}

    def is_reachable(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=HEALTH_TIMEOUT)
        r.raise_for_status()
        return r.json()

    def stats(self, session_id: str | None = None) -> Dict[str, Any]:
        params = {"session_id": session_id} if session_id else {}
        r = requests.get(f"{self.base_url}/stats", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def upload(
        self,
        files: List[Tuple[str, bytes]],
        session_id: str | None = None,
        reindex: bool = False,
    ) -> Dict[str, Any]:
        multipart = [
            ("files", (name, data, "application/pdf")) for name, data in files
        ]
        params: Dict[str, Any] = {}
        if session_id:
            params["session_id"] = session_id
        if reindex:
            params["reindex"] = "true"
        r = requests.post(
            f"{self.base_url}/upload",
            files=multipart,
            params=params,
            timeout=TIMEOUT,
        )
        if not r.ok:
            raise APIError(_error_detail(r), r.status_code)
        return r.json()

    def chat(
        self,
        question: str,
        *,
        use_hybrid: bool = False,
        source_filter: str | None = None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "question": question,
            "use_hybrid": use_hybrid,
            "source_filter": source_filter,
            "session_id": session_id,
            "stream": False,
        }
        r = requests.post(
            f"{self.base_url}/chat",
            json=payload,
            timeout=TIMEOUT,
        )
        if not r.ok:
            raise APIError(_error_detail(r), r.status_code)
        return r.json()

    def chat_stream_tokens(
        self,
        question: str,
        *,
        use_hybrid: bool = False,
        source_filter: str | None = None,
        session_id: str | None = None,
    ) -> Generator[str, None, Dict[str, Any]]:
        """Yield LLM tokens; meta dict is attached on the generator after completion."""
        payload = {
            "question": question,
            "use_hybrid": use_hybrid,
            "source_filter": source_filter,
            "session_id": session_id,
            "stream": True,
        }
        meta: Dict[str, Any] = {}
        with requests.post(
            f"{self.base_url}/chat/stream",
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        ) as r:
            if not r.ok:
                raise APIError(_error_detail(r), r.status_code)
            event: str | None = None
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event == "meta":
                        meta = data
                    elif event == "done":
                        break
                    elif "token" in data:
                        yield data["token"]
        self.last_stream_meta = meta

    def reset(self, session_id: str | None = None) -> None:
        params = {"session_id": session_id} if session_id else {}
        r = requests.delete(f"{self.base_url}/reset", params=params, timeout=60)
        r.raise_for_status()


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = body.get("detail", response.text)
        if isinstance(detail, list):
            return "; ".join(str(d) for d in detail)
        return str(detail)
    except Exception:
        return response.text or f"HTTP {response.status_code}"
