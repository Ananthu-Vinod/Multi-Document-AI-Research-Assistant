"""Expand user queries for acronym and keyword matching."""

import re
from typing import Dict, List, Set

from ingestion.aliases import acronym_from_filename, extract_query_acronyms, infer_document_aliases
from langchain_core.documents import Document
from logger import get_logger

logger = get_logger(__name__)


def _aliases_for_source(source: str, sample_text: str, metadata_aliases: str) -> List[str]:
    if metadata_aliases:
        return [a.strip() for a in metadata_aliases.split("|") if a.strip()]
    return infer_document_aliases(source, sample_text)


def build_alias_map(documents: List[Document]) -> Dict[str, Set[str]]:
    """
    Map acronyms to related phrases using document metadata and filenames.

    Example: AMBD -> {'Applied Model Based Design', ...}
    """
    by_source: Dict[str, List[str]] = {}

    for doc in documents:
        source = doc.metadata.get("source", "")
        if not source or source in by_source:
            continue
        by_source[source] = _aliases_for_source(
            source,
            doc.page_content[:2500],
            doc.metadata.get("aliases", ""),
        )

    alias_map: Dict[str, Set[str]] = {}
    for source, alias_list in by_source.items():
        acronyms = [
            a.upper()
            for a in alias_list
            if len(a) <= 12 and a.replace("-", "").isalnum() and a.upper() == a
        ]
        titles = [a for a in alias_list if a.upper() not in acronyms]

        filename_code = acronym_from_filename(source)
        if filename_code:
            acronyms.append(filename_code)

        for ac in set(acronyms):
            alias_map.setdefault(ac, set()).update(titles)

    return alias_map


def expand_query(query: str, alias_map: Dict[str, Set[str]]) -> str:
    """
    Append related phrases when the user asks about a known acronym.

    Example: 'what is AMBD' -> 'what is AMBD Applied Model Based Design ...'
    """
    expanded_terms: List[str] = []
    for acronym in extract_query_acronyms(query):
        related = alias_map.get(acronym.upper(), set())
        if related:
            expanded_terms.extend(sorted(related))
            logger.info("Expanded acronym %s with: %s", acronym, related)

    if not expanded_terms:
        return query

    extra = " ".join(dict.fromkeys(expanded_terms))
    return f"{query} {extra}"


def should_use_hybrid(query: str) -> bool:
    """Use BM25 for acronym-heavy or short keyword queries."""
    if extract_query_acronyms(query):
        return True
    tokens = re.findall(r"\b\w+\b", query.lower())
    return len(tokens) <= 6
