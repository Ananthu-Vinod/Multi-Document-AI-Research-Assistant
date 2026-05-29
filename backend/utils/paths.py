"""Path resolution utilities."""

from pathlib import Path

from config import Config


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """
    Resolve a path relative to BASE_DIR unless already absolute.

    Args:
        path: Relative or absolute path
        base: Optional base directory (defaults to Config.BASE_DIR)

    Returns:
        Resolved absolute Path
    """
    base = base or Config.BASE_DIR
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()
