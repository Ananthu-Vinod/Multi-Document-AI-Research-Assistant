"""Sidebar: settings, upload, database controls."""

import streamlit as st

from components.api_client import APIError, RAGApiClient
from components.local_client import LocalRAGClient
from components.rag_client import ClientType, create_rag_client


def render_sidebar(client: ClientType) -> None:
    with st.sidebar:
        st.markdown("### Ask My Docs")
        st.caption("RAG over your PDFs")

        st.markdown("---")
        st.markdown("**Connection**")

        mode = st.session_state.get("rag_mode", "local")
        force_local = st.checkbox(
            "Embedded mode (no API server)",
            value=mode == "local",
            help="Run RAG in-process. Uncheck to use FastAPI at the URL below.",
        )

        api_url_default = (
            client.base_url
            if isinstance(client, RAGApiClient)
            else st.session_state.get("api_url", "http://localhost:8000")
        )
        backend_url = st.text_input(
            "API URL",
            value=api_url_default,
            disabled=force_local,
            help="FastAPI server when not in embedded mode",
        )
        st.session_state.api_url = backend_url

        if force_local:
            if not isinstance(st.session_state.rag_client, LocalRAGClient):
                c, m = create_rag_client(
                    session_id=st.session_state.session_id, prefer_local=True
                )
                st.session_state.rag_client = c
                st.session_state.rag_mode = m
        else:
            api = RAGApiClient(base_url=backend_url)
            if api.is_reachable():
                st.session_state.rag_client = api
                st.session_state.rag_mode = "api"
            else:
                c, m = create_rag_client(
                    session_id=st.session_state.session_id, prefer_local=True
                )
                st.session_state.rag_client = c
                st.session_state.rag_mode = m

        client = st.session_state.rag_client
        mode = st.session_state.rag_mode

        try:
            health = client.health()
            if mode == "local":
                st.success(
                    f"Embedded RAG · {health.get('chunk_count', 0)} chunks indexed"
                )
            else:
                st.success(f"API online · {health.get('chunk_count', 0)} chunks")
        except Exception as exc:
            st.error(f"Connection failed: {exc}")
            if not force_local:
                st.info(
                    "Enable **Embedded mode** above, or start the API:\n\n"
                    "`cd backend` → `uvicorn app:app --port 8000`"
                )

        if mode == "local" and not force_local:
            st.caption("API was offline — using embedded RAG automatically.")

        st.markdown("---")
        st.markdown("**Search**")
        st.session_state.use_hybrid = st.checkbox(
            "Hybrid search (BM25 + vector)",
            value=st.session_state.get("use_hybrid", False),
        )
        st.session_state.enable_streaming = st.checkbox(
            "Stream responses",
            value=st.session_state.get("enable_streaming", True),
        )
        st.session_state.source_filter = st.text_input(
            "Source filter",
            value=st.session_state.get("source_filter", ""),
            placeholder="report.pdf",
        )

        st.markdown("---")
        st.markdown("**Documents**")
        uploaded = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        reindex = st.checkbox("Re-index (clear DB first)", value=False)
        if uploaded and st.button("Process PDFs", type="primary", use_container_width=True):
            _process_uploads(st.session_state.rag_client, uploaded, reindex)

        if st.button("Reset index", type="secondary", use_container_width=True):
            try:
                st.session_state.rag_client.reset(
                    session_id=st.session_state.session_id
                )
                st.session_state.messages = []
                st.session_state.documents_ready = False
                st.success("Index cleared")
                st.rerun()
            except APIError as exc:
                st.error(str(exc))

        st.markdown("---")
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


def _process_uploads(client: ClientType, files, reindex: bool) -> None:
    with st.spinner("Indexing PDFs… This may take a minute on first run."):
        try:
            payloads = [(f.name, f.getvalue()) for f in files]
            result = client.upload(
                payloads,
                session_id=st.session_state.session_id,
                reindex=reindex,
            )
            st.session_state.documents_ready = result.get("total_chunks", 0) > 0
            if result.get("chunks_added", 0) > 0:
                st.success(
                    f"Added {result['chunks_added']} chunks from "
                    f"{result['files_processed']} file(s). "
                    f"Total: {result['total_chunks']}"
                )
            elif result.get("message"):
                st.warning(result["message"])
            else:
                st.warning("No new content indexed.")
        except APIError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
