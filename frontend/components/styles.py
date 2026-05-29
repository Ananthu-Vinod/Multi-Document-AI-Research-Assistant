"""Inject custom CSS into Streamlit."""

from pathlib import Path

import streamlit as st


def load_dark_theme() -> None:
    css_path = Path(__file__).resolve().parents[1] / "styles" / "dark_theme.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
