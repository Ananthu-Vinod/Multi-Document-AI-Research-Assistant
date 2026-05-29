"""
Ask My Docs — Streamlit frontend.

Run from project root:
    streamlit run frontend/streamlit_app.py

Or from frontend/:
    streamlit run streamlit_app.py

Uses FastAPI when available; otherwise embedded local RAG (no separate server).
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
    page_title="Ask My Docs",
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

    st.markdown('<p class="app-title">Ask My Docs</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Upload PDFs and chat with citations — powered by hybrid RAG.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        st.session_state.show_sources = st.toggle(
            "Show sources",
            value=st.session_state.get("show_sources", True),
        )

    st.markdown("---")
    render_chat(st.session_state.rag_client)


if __name__ == "__main__":
    main()
