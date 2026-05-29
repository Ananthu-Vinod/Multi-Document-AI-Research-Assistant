"""Document ingestion layer."""

from .deduplication import DocumentDeduplicator
from .processor import DocumentProcessor

__all__ = ["DocumentProcessor", "DocumentDeduplicator"]
