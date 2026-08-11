"""Format retrieval results and clean answer typography into human-readable output."""

import re
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


def clean_markdown_text(text: str) -> str:
    """Post-process LLM response text to replace raw markdown bullet asterisks with clean unicode bullets."""
    if not text:
        return ""
    lines = []
    for line in text.split("\n"):
        # Match leading whitespace followed by * or - bullet marker
        match = re.match(r"^(\s*)[\*\-]\s+(.*)$", line)
        if match:
            indent, content = match.groups()
            line = f"{indent}• {content}"
        lines.append(line)
    return "\n".join(lines)


def format_citation(metadata: Dict[str, Any]) -> str:
    """Build a single citation line like 'report.pdf (Page 5)'."""
    source = metadata.get("source") or metadata.get("display_name") or "Unknown"
    page = metadata.get("page")
    if page is not None:
        return f"{source} (Page {page})"
    return str(source)


def chunks_from_results(
    results: List[Tuple[Document, float]],
    *,
    preview_chars: int = 500,
) -> List[Dict[str, Any]]:
    """Serialize retrieved chunks for API responses."""
    chunks: List[Dict[str, Any]] = []
    for doc, score in results:
        meta = dict(doc.metadata or {})
        content = doc.page_content
        chunks.append(
            {
                "content": content,
                "preview": (
                    content[:preview_chars] + "..."
                    if len(content) > preview_chars
                    else content
                ),
                "score": round(float(score), 6),
                "source": meta.get("source") or meta.get("display_name"),
                "page": meta.get("page"),
                "metadata": meta,
                "citation": format_citation(meta),
            }
        )
    return chunks


def unique_citations(chunks: List[Dict[str, Any]]) -> List[str]:
    """Deduplicated citation lines preserving order."""
    seen: set[str] = set()
    citations: List[str] = []
    for chunk in chunks:
        cite = chunk.get("citation")
        if cite and cite not in seen:
            seen.add(cite)
            citations.append(cite)
    return citations
