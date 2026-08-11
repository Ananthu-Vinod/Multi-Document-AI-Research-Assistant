"""
Ask My Docs — Premium RAG Application Streamlit Frontend.
"""

import sys
import uuid
from pathlib import Path

import streamlit as st

_FRONTEND = Path(__file__).resolve().parent
_ROOT = _FRONTEND.parent
for path in (_FRONTEND, _ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from components.chat import render_chat  # noqa: E402
from components.rag_client import create_rag_client  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402
from components.styles import load_dark_theme  # noqa: E402

st.set_page_config(
    page_title="Ask My Docs | AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "documents_ready": False,
        "use_hybrid": False,
        "enable_streaming": True,
        "source_filter": "",
        "show_sources": True,
        "rag_mode": "local",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "rag_client" not in st.session_state:
        client, mode = create_rag_client(session_id=st.session_state.session_id)
        st.session_state.rag_client = client
        st.session_state.rag_mode = mode

    try:
        stats = st.session_state.rag_client.stats(
            session_id=st.session_state.session_id
        )
        if stats.get("chunk_count", 0) > 0:
            st.session_state.documents_ready = True
    except Exception:
        pass


def main() -> None:
    load_dark_theme()
    _init_state()

    render_sidebar(st.session_state.rag_client)

    # Hero Header Section
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">Ask My Docs</h1>
            <p class="hero-sub">Upload multi-page PDFs, query with hybrid RAG, and get grounded answers with page-level citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If no conversation started yet, render feature highlight cards
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Hybrid BM25 + Vector Search</div>
                    <div class="feature-desc">Fuses dense semantic embeddings with exact keyword matching and query acronym expansion.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📌</div>
                    <div class="feature-title">Page-Level Citations</div>
                    <div class="feature-desc">Every AI answer references exact source files and page numbers for complete auditability.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔐</div>
                    <div class="feature-title">JWT Auth & Postgres DB</div>
                    <div class="feature-desc">Secured with user authentication, persistent PostgreSQL metadata, and rate limiting.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Header control row
    top_col1, top_col2 = st.columns([3, 1])
    with top_col2:
        st.session_state.show_sources = st.toggle(
            "Show sources & relevance scores",
            value=st.session_state.get("show_sources", True),
        )

    st.markdown("---")
    render_chat(st.session_state.rag_client)


if __name__ == "__main__":
    main()
