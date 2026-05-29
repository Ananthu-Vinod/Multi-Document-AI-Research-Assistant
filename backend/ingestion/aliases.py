"""Infer document aliases and titles for acronym-aware retrieval."""

import re
from pathlib import Path
from typing import List


_INTRO_PATTERN = re.compile(
    r"introduction\s+to\s+(.{3,100}?)(?:\s+(?:dr|prof|mr|ms)\b|[\n.]|$)",
    re.IGNORECASE,
)
_ACRONYM_IN_QUERY = re.compile(r"\b([A-Z]{2,10})\b")


def acronym_from_filename(filename: str) -> str | None:
    """
    Extract likely acronym from filenames like Unit_1__AMBD.pdf.

    Uses the segment after the last double-underscore.
    """
    stem = Path(filename).stem
    if "__" in stem:
        code = stem.rsplit("__", 1)[-1].strip()
        if 2 <= len(code) <= 12 and code.replace("-", "").isalnum():
            return code.upper()
    return None


def title_from_text(text: str) -> str | None:
    """Parse 'Introduction to ...' course titles from page text."""
    match = _INTRO_PATTERN.search(text)
    if match:
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
        title = re.split(r"\s+(?:dr|prof)\b", title, maxsplit=1, flags=re.I)[0].strip()
        if len(title) >= 5:
            return title
    return None


def infer_document_aliases(filename: str, sample_text: str = "") -> List[str]:
    """
    Build searchable aliases for a document.

    Example: Unit_1__AMBD.pdf + intro text ->
        ['AMBD', 'Applied Model Based Design']
    """
    aliases: List[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        if not value or value.lower() in seen:
            return
        seen.add(value.lower())
        aliases.append(value)

    code = acronym_from_filename(filename)
    if code:
        add(code)

    title = title_from_text(sample_text)
    if title:
        add(title)

    return aliases


def extract_query_acronyms(query: str) -> List[str]:
    """Find uppercase tokens that look like acronyms in the user question."""
    tokens = _ACRONYM_IN_QUERY.findall(query)
    # Filter common false positives
    stop = {"PDF", "API", "URL", "HTTP", "HTTPS", "JSON", "XML", "HTML"}
    return [t for t in tokens if t not in stop]
