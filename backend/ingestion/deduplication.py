"""Document deduplication using content hashes."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from logger import get_logger
from utils.files import compute_file_hash

logger = get_logger(__name__)


class DocumentDeduplicator:
    """Tracks indexed documents by SHA-256 hash to prevent duplicate ingestion."""

    def __init__(self, registry_path: Path | None = None):
        self.registry_path = registry_path or Config.DOCUMENT_REGISTRY_PATH
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, dict] = self._load_registry()

    def _load_registry(self) -> Dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load document registry: %s", exc)
            return {}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(self._registry, handle, indent=2)

    def is_indexed(self, file_path: Path) -> bool:
        """Return True if file hash already exists in registry."""
        file_hash = compute_file_hash(file_path)
        return file_hash in self._registry

    def get_record(self, file_path: Path) -> Optional[dict]:
        """Get registry record for a file if indexed."""
        file_hash = compute_file_hash(file_path)
        return self._registry.get(file_hash)

    def register(
        self,
        file_path: Path,
        *,
        original_name: str,
        chunk_count: int,
        session_id: str | None = None,
    ) -> str:
        """
        Register a newly indexed document.

        Returns:
            Document hash
        """
        file_hash = compute_file_hash(file_path)
        self._registry[file_hash] = {
            "hash": file_hash,
            "original_name": original_name,
            "stored_path": str(file_path),
            "chunk_count": chunk_count,
            "session_id": session_id,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_registry()
        logger.info("Registered document hash=%s name=%s", file_hash[:12], original_name)
        return file_hash

    def filter_new_files(self, file_paths: List[Path]) -> tuple[List[Path], List[Path]]:
        """
        Split files into new (not indexed) and duplicate (already indexed).

        Returns:
            (new_files, duplicate_files)
        """
        new_files: List[Path] = []
        duplicates: List[Path] = []

        for path in file_paths:
            if self.is_indexed(path):
                duplicates.append(path)
                logger.info("Skipping duplicate document: %s", path.name)
            else:
                new_files.append(path)

        return new_files, duplicates

    def clear(self) -> None:
        """Clear registry (e.g., on database reset)."""
        self._registry = {}
        if self.registry_path.exists():
            self.registry_path.unlink()
        logger.info("Document registry cleared")
