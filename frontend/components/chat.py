"""ChatGPT-style chat UI with rich starter prompts and source citation panels."""

import streamlit as st

from components.api_client import APIError
from components.citations import render_chunk_panel, render_citations
from components.rag_client import ClientType


def render_chat(client: ClientType) -> None:
    # If no documents uploaded yet, render guidance hero panel
    if not st.session_state.get("documents_ready"):
        st.markdown(
            """
            <div class="upload-callout">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📂</div>
                <h3>No Documents Uploaded Yet</h3>
                <p>Upload one or more PDF files in the sidebar on the left and click <b>Process PDFs</b> to build your interactive knowledge base.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # If documents are ready but no messages exist yet, show starter prompt pills
    if not st.session_state.messages:
        st.markdown('<p class="quick-prompts-label">Suggested Questions</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💡 Summarize core findings", use_container_width=True):
                st.session_state.pending_prompt = "Summarize the core findings and main points of the uploaded documents."
                st.rerun()
            if st.button("📊 Extract key metrics & data", use_container_width=True):
                st.session_state.pending_prompt = "Extract all key metrics, statistics, tables, and data points discussed."
                st.rerun()
        with col2:
            if st.button("🔍 What are the key methodology steps?", use_container_width=True):
                st.session_state.pending_prompt = "What are the key methodology steps, frameworks, or procedures outlined?"
                st.rerun()
            if st.button("⚡ List major conclusions & recommendations", use_container_width=True):
                st.session_state.pending_prompt = "List the major conclusions, takeaways, and recommendations."
                st.rerun()

    # Render previous conversation history
    for msg in st.session_state.messages:
        _render_message(msg)

    # Check if a starter prompt button was clicked
    prompt = None
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        prompt = st.chat_input("Ask a question about your documents…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        _render_message(st.session_state.messages[-1])
        _handle_assistant_turn(client, prompt)


def _render_message(msg: dict) -> None:
    role = msg["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "📚"):
        st.markdown(msg.get("content", ""))
        if role == "assistant":
            if msg.get("citations"):
                render_citations(msg["citations"])
            if msg.get("chunks") and st.session_state.get("show_sources", True):
                with st.expander("Sources & relevance scores", expanded=False):
                    render_chunk_panel(msg["chunks"])
            if msg.get("latency_ms") is not None:
                st.caption(
                    f"{msg.get('search_mode', 'vector')} search · "
                    f"{msg['latency_ms']:.0f} ms"
                )


def _handle_assistant_turn(client: ClientType, prompt: str) -> None:
    source_filter = (
        st.session_state.source_filter.strip()
        if st.session_state.get("source_filter")
        else None
    )
    use_hybrid = st.session_state.get("use_hybrid", False)
    session_id = st.session_state.session_id
    use_stream = st.session_state.get("enable_streaming", True)

    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message("assistant", avatar="📚"):
            try:
                if use_stream:
                    answer, meta = _stream_answer(
                        client, prompt, use_hybrid, source_filter, session_id
                    )
                else:
                    with st.spinner("Searching documents…"):
                        data = client.chat(
                            prompt,
                            use_hybrid=use_hybrid,
                            source_filter=source_filter,
                            session_id=session_id,
                        )
                    answer = data.get("answer") or "_No answer generated._"
                    meta = data

                assistant_msg = {
                    "role": "assistant",
                    "content": answer,
                    "citations": meta.get("citations", []),
                    "chunks": meta.get("chunks", []),
                    "latency_ms": meta.get("latency_ms"),
                    "search_mode": meta.get("search_mode"),
                }
                st.session_state.messages.append(assistant_msg)
            except APIError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

    placeholder.empty()
    st.rerun()


def _stream_answer(
    client: ClientType,
    prompt: str,
    use_hybrid: bool,
    source_filter: str | None,
    session_id: str,
) -> tuple[str, dict]:
    def token_iter():
        yield from client.chat_stream_tokens(
            prompt,
            use_hybrid=use_hybrid,
            source_filter=source_filter,
            session_id=session_id,
        )

    with st.spinner("Retrieving context & generating response…"):
        try:
            streamed = st.write_stream(token_iter)
            answer = streamed if isinstance(streamed, str) else ""
            meta = dict(client.last_stream_meta)
            return answer or "_No answer generated._", meta
        except APIError:
            raise
        except Exception:
            data = client.chat(
                prompt,
                use_hybrid=use_hybrid,
                source_filter=source_filter,
                session_id=session_id,
            )
            return data.get("answer") or "", data
