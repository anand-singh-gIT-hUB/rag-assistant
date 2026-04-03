"""
streamlit_app/components/uploader.py
"""
import streamlit as st
from streamlit_app.utils import api_client


def render_uploader():
    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt", "md"],
        help="Supported: PDF, DOCX, TXT, Markdown",
    )
    if uploaded and st.button("📤 Index Document", type="primary"):
        with st.spinner(f"Indexing `{uploaded.name}`…"):
            try:
                result = api_client.upload_document(
                    file_bytes=uploaded.read(),
                    filename=uploaded.name,
                )
                st.success(
                    f"✅ Indexed **{result['file_name']}** — "
                    f"{result['num_chunks']} chunks stored."
                )
            except Exception as e:
                st.error(f"Upload failed: {e}")
