"""Citation and source chunk display."""

import streamlit as st


def render_citations(citations: list[str]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    pills = " ".join(
        f'<span class="citation-pill">{c}</span>' for c in citations
    )
    st.markdown(pills, unsafe_allow_html=True)


def render_chunk_panel(chunks: list[dict]) -> None:
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0)
        cite = chunk.get("citation", chunk.get("source", "Unknown"))
        preview = chunk.get("preview", chunk.get("content", ""))[:600]
        st.markdown(
            f'<div class="chunk-card">'
            f'<strong>#{i}</strong> <span class="score-badge">score {score:.4f}</span><br/>'
            f'<em>{cite}</em><br/>'
            f'<pre style="white-space:pre-wrap;font-size:0.85rem;">{preview}</pre>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Simple score bar for retrieval visualization
        st.progress(min(max(float(score), 0.0), 1.0))
