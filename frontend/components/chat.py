"""ChatGPT-style chat UI."""

import streamlit as st

from components.api_client import APIError
from components.citations import render_chunk_panel, render_citations
from components.rag_client import ClientType


def render_chat(client: ClientType) -> None:
    if not st.session_state.get("documents_ready"):
        st.info("Upload and process PDFs in the sidebar to start chatting.")
        return

    for msg in st.session_state.messages:
        _render_message(msg)

    if prompt := st.chat_input("Ask a question about your documents…"):
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
                with st.expander("Sources & scores", expanded=False):
                    render_chunk_panel(msg["chunks"])
            if msg.get("latency_ms") is not None:
                st.caption(
                    f"{msg.get('search_mode', 'vector')} · "
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
                    with st.spinner("Searching…"):
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

    with st.spinner("Retrieving context…"):
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
