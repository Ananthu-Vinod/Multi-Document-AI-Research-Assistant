"""Context string builder for LLM prompts."""

from typing import Dict, List

from utils.tokens import truncate_to_char_budget


class ContextBuilder:
    """Formats retrieved chunks into a bounded context string."""

    @staticmethod
    def build(
        context_chunks: List[str],
        metadata: List[Dict] | None = None,
        max_chars: int | None = None,
    ) -> str:
        """
        Build formatted context with source citations.

        Args:
            context_chunks: Retrieved text chunks
            metadata: Parallel metadata for citations
            max_chars: Optional per-chunk char cap

        Returns:
            Formatted context block
        """
        parts: List[str] = []
        for i, chunk in enumerate(context_chunks):
            text = chunk
            if max_chars:
                text = truncate_to_char_budget(chunk, max_chars // max(len(context_chunks), 1))

            parts.append(f"[Context {i + 1}]")
            parts.append(text)

            if metadata and i < len(metadata):
                source = metadata[i].get("source", "Unknown")
                page = metadata[i].get("page", "N/A")
                parts.append(f"(Source: {source}, Page: {page})")
            parts.append("")

        return "\n".join(parts).strip()
