"""Token and character budgeting utilities."""

from typing import List, Tuple

from config import Config


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses tiktoken when available; falls back to character-based estimate.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // Config.CHARS_PER_TOKEN_ESTIMATE)


def truncate_to_char_budget(text: str, max_chars: int) -> str:
    """Truncate text to a maximum character count."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate text to an approximate token budget."""
    max_chars = max_tokens * Config.CHARS_PER_TOKEN_ESTIMATE
    return truncate_to_char_budget(text, max_chars)


def pack_chunks_by_budget(
    chunks: List[Tuple[str, float, dict]],
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> List[Tuple[str, float, dict]]:
    """
    Pack ranked chunks into a context budget, highest score first.

    Args:
        chunks: List of (content, score, metadata) sorted by relevance desc
        max_chars: Character budget (defaults to Config.MAX_CONTEXT_CHARS)
        max_tokens: Token budget (defaults to Config.MAX_CONTEXT_TOKENS)

    Returns:
        Subset of chunks that fits within budget
    """
    max_chars = max_chars or Config.MAX_CONTEXT_CHARS
    max_tokens = max_tokens or Config.MAX_CONTEXT_TOKENS

    packed: List[Tuple[str, float, dict]] = []
    used_chars = 0
    used_tokens = 0

    for content, score, metadata in chunks:
        chunk_chars = len(content)
        chunk_tokens = estimate_tokens(content)

        if used_chars + chunk_chars > max_chars:
            continue
        if used_tokens + chunk_tokens > max_tokens:
            continue

        packed.append((content, score, metadata))
        used_chars += chunk_chars
        used_tokens += chunk_tokens

    return packed
