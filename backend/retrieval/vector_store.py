"""ChromaDB vector storage with metadata filtering and similarity conversion."""

from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Config
from embeddings.generator import get_embedding_generator
from logger import get_logger
from retrieval.scoring import distance_to_similarity

logger = get_logger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain embeddings wrapper using the shared singleton model."""

    def __init__(self):
        self._generator = None

    def _get_generator(self):
        if self._generator is None:
            self._generator = get_embedding_generator()
        return self._generator

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_generator().generate_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_generator().generate_single_embedding(text)


class VectorStore:
    """Manages ChromaDB persistence and similarity search."""

    def __init__(self, collection_name: str | None = None):
        Config.ensure_directories()
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embeddings = SentenceTransformerEmbeddings()
        self.client = chromadb.PersistentClient(path=str(Config.CHROMA_PERSIST_DIR))
        self.collection = None
        self.vectorstore: Chroma | None = None
        logger.info("ChromaDB initialized at %s", Config.CHROMA_PERSIST_DIR)

    def collection_exists(self) -> bool:
        """Check whether the configured collection already exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except NotFoundError:
            return False
        except Exception as exc:
            logger.warning("Could not verify collection existence: %s", exc)
            return False

    def create_collection(self, collection_name: str | None = None) -> None:
        """Create or load a Chroma collection."""
        name = collection_name or self.collection_name
        self.collection_name = name

        try:
            self.collection = self.client.get_collection(name=name)
            logger.info("Loaded existing collection: %s", name)
        except NotFoundError:
            self.collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created new collection: %s", name)

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add document chunks to the vector store."""
        if self.vectorstore is None:
            self.create_collection()

        if not documents:
            logger.warning("No documents provided for indexing")
            return

        try:
            self.vectorstore.add_documents(documents)
            count = self.collection.count() if self.collection else len(documents)
            logger.info("Added %d chunks (total=%d)", len(documents), count)
        except Exception as exc:
            logger.error("Failed to add documents: %s", exc)
            raise RuntimeError(f"Vector store indexing failed: {exc}") from exc

    def _build_chroma_filter(
        self, filter_metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not filter_metadata:
            return None
        # Chroma expects {"$and": [{"field": {"$eq": value}}, ...]} for multiple filters
        clauses = []
        for key, value in filter_metadata.items():
            if value is not None:
                clauses.append({key: {"$eq": value}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        *,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Semantic search without scores."""
        results = self.similarity_search_with_scores(
            query,
            k=k,
            filter_metadata=filter_metadata,
            convert_distance=True,
        )
        return [doc for doc, _ in results]

    def similarity_search_with_scores(
        self,
        query: str,
        k: int | None = None,
        *,
        filter_metadata: Optional[Dict[str, Any]] = None,
        convert_distance: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Search with scores where higher values indicate better matches.

        When convert_distance=True, Chroma distances are mapped to similarity.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_collection() first.")

        k = k or Config.TOP_K_RESULTS
        chroma_filter = self._build_chroma_filter(filter_metadata)

        try:
            if chroma_filter:
                raw = self.vectorstore.similarity_search_with_score(
                    query, k=k, filter=chroma_filter
                )
            else:
                raw = self.vectorstore.similarity_search_with_score(query, k=k)

            results: List[Tuple[Document, float]] = []
            for doc, score in raw:
                final_score = (
                    distance_to_similarity(score) if convert_distance else score
                )
                results.append((doc, final_score))

            results.sort(key=lambda item: item[1], reverse=True)
            logger.info("Retrieved %d chunks for query", len(results))
            return results

        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise RuntimeError(f"Vector search failed: {exc}") from exc

    def delete_collection(self) -> None:
        """Delete the active collection."""
        if self.collection:
            name = self.collection.name
            self.client.delete_collection(name=name)
            logger.info("Deleted collection: %s", name)
        self.collection = None
        self.vectorstore = None

    def get_collection_stats(self) -> int:
        """Return chunk count for the active collection."""
        if self.collection:
            return self.collection.count()
        return 0

    def load_existing(self) -> bool:
        """Attach to an existing persisted collection if available."""
        if not self.collection_exists():
            return False
        self.create_collection()
        return self.vectorstore is not None

    def get_all_documents(self, limit: int | None = None) -> List[Document]:
        """Load all stored chunks from Chroma for in-memory index rebuild."""
        if self.collection is None:
            return []

        try:
            result = self.collection.get(limit=limit)
            documents: List[Document] = []
            texts = result.get("documents") or []
            metadatas = result.get("metadatas") or [{}] * len(texts)

            for text, metadata in zip(texts, metadatas):
                if text:
                    documents.append(
                        Document(page_content=text, metadata=metadata or {})
                    )
            return documents
        except Exception as exc:
            logger.warning("Could not load documents from collection: %s", exc)
            return []
