"""Unit tests for BM25 document length handling."""


from langchain_core.documents import Document
from retrieval.bm25 import BM25Retriever


def test_bm25_uses_total_token_count_not_unique_vocab():
    """Document length should use total tokens, not unique token count."""
    docs = [
        Document(page_content="test test test test", metadata={}),
        Document(page_content="one two three", metadata={}),
    ]
    retriever = BM25Retriever()
    retriever.index_documents(docs)

    assert retriever.doc_lengths[0] == 4
    assert retriever.doc_lengths[1] == 3
    assert retriever.avg_doc_len == 3.5


if __name__ == "__main__":
    test_bm25_uses_total_token_count_not_unique_vocab()
    print("test_bm25_lengths: PASS")
