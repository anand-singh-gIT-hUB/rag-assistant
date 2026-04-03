"""
streamlit_app/components/source_viewer.py
"""
import streamlit as st
from streamlit_app.utils import api_client


def render_source_debug(question: str, top_k: int = 10):
    with st.expander("🔍 Debug: Raw Retrieved Chunks", expanded=False):
        if st.button("Fetch raw chunks"):
            with st.spinner("Retrieving…"):
                try:
                    result = api_client.retrieve(question=question, top_k=top_k)
                    chunks = result.get("chunks", [])
                    if not chunks:
                        st.info("No chunks returned.")
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}** — `{c['file_name']}` | score: `{c['score']:.4f}`")
                        st.text(c["text"][:400])
                        st.divider()
                except Exception as e:
                    st.error(f"Error: {e}")
