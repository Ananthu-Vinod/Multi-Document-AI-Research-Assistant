"""Secure file handling utilities."""

import hashlib
import re
import uuid
from pathlib import Path

from config import Config
from logger import get_logger

logger = get_logger(__name__)

_UNSAFE_FILENAME_PATTERN = re.compile(r"[^\w.\-]", re.UNICODE)


def sanitize_filename(filename: str, default_ext: str = ".pdf") -> str:
    """
    Sanitize an uploaded filename to prevent path traversal and unsafe names.

    Args:
        filename: Original uploaded filename
        default_ext: Extension to use if missing after sanitization

    Returns:
        Safe basename-only filename
    """
    # Strip directory components (path traversal defense)
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        name = f"{uuid.uuid4().hex}{default_ext}"

    stem = Path(name).stem
    suffix = Path(name).suffix or default_ext
    safe_stem = _UNSAFE_FILENAME_PATTERN.sub("_", stem).strip("._") or uuid.uuid4().hex
    safe_name = f"{safe_stem}{suffix}"

    # Final guard: must not contain path separators
    if "/" in safe_name or "\\" in safe_name:
        safe_name = f"{uuid.uuid4().hex}{default_ext}"

    return safe_name


def safe_upload_path(filename: str, uploads_dir: Path | None = None) -> Path:
    """
    Build a safe absolute path for an uploaded file.

    Args:
        filename: Original uploaded filename
        uploads_dir: Target uploads directory

    Returns:
        Safe resolved path under uploads directory
    """
    uploads_dir = uploads_dir or Config.UPLOADS_DIR
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(filename)
    target = (uploads_dir / safe_name).resolve()
    uploads_root = uploads_dir.resolve()

    # Prevent path traversal outside uploads directory
    if uploads_root not in target.parents and target != uploads_root:
        logger.warning("Path traversal attempt blocked for filename: %s", filename)
        target = uploads_root / f"{uuid.uuid4().hex}.pdf"

    return target


def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 hash of a file for deduplication.

    Args:
        file_path: Path to file
        chunk_size: Read chunk size in bytes

    Returns:
        Hex digest of file contents
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
