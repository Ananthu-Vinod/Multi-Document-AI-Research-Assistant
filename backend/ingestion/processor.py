"""Document processing module for PDF ingestion and chunking."""

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from ingestion.aliases import infer_document_aliases
from logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Processes PDF documents for the RAG pipeline."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, pdf_path: str | Path) -> List[Document]:
        """
        Load a PDF file and extract text per page.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page-level Document objects
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            loader = PyPDFLoader(str(path))
            documents = loader.load()
            if not documents:
                logger.warning("No text extracted from PDF: %s", path)
            else:
                logger.info("Loaded %d pages from %s", len(documents), path.name)
            return documents
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", path, exc)
            raise ValueError(f"Could not read PDF '{path.name}': {exc}") from exc

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into retrieval-sized chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info("Split into %d chunks", len(chunks))
            return chunks
        except Exception as exc:
            logger.error("Failed to split documents: %s", exc)
            raise

    def process_pdf(
        self,
        pdf_path: str | Path,
        *,
        session_id: str | None = None,
        document_hash: str | None = None,
        display_name: str | None = None,
    ) -> List[Document]:
        """
        Full pipeline: load PDF and split into chunks with citation metadata.

        Args:
            pdf_path: Path to PDF
            session_id: Optional session identifier for metadata filtering
            document_hash: Optional SHA-256 hash for deduplication tracking
            display_name: Human-readable source name for citations

        Returns:
            Chunked Document objects
        """
        path = Path(pdf_path)
        logger.info("Processing PDF: %s", path.name)

        documents = self.load_pdf(path)
        if not documents:
            return []

        chunks = self.split_documents(documents)
        source_name = display_name or path.name
        sample_text = " ".join(doc.page_content for doc in documents[:3])
        aliases = infer_document_aliases(source_name, sample_text)
        alias_str = "|".join(aliases) if aliases else ""

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_id": i,
                    "source": source_name,
                    "file_path": str(path.resolve()),
                    "session_id": session_id,
                    "document_hash": document_hash,
                    "aliases": alias_str,
                }
            )
            # Prefix improves embedding + BM25 for filename acronyms (e.g. AMBD)
            if aliases:
                prefix = f"[Document aliases: {', '.join(aliases)}]\n"
                chunk.page_content = prefix + chunk.page_content

        if aliases:
            logger.info("Document aliases for %s: %s", source_name, aliases)

        return chunks
