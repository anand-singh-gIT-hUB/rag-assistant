"""
streamlit_app/pages/2_Documents.py
────────────────────────────────────
Document management: upload, list, delete, reindex.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Documents — RAG Assistant", page_icon="📂", layout="wide")

from streamlit_app.utils.session import init_session
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.uploader import render_uploader
from streamlit_app.utils import api_client

init_session()
render_sidebar()

st.title("📂 Documents")

# ── Upload ────────────────────────────────────────────────────────────────────
st.subheader("Upload New Document")
render_uploader()

st.divider()

# ── Document list ─────────────────────────────────────────────────────────────
st.subheader("Indexed Documents")
if st.button("🔄 Refresh list"):
    st.rerun()

try:
    data = api_client.list_documents()
    docs = data.get("documents", [])
    if not docs:
        st.info("No documents indexed yet. Upload one above!")
    else:
        for doc in docs:
            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 2, 2])
                with c1:
                    st.markdown(f"**{doc['file_name']}**")
                    st.caption(
                        f"ID: `{doc['doc_id']}` · {doc['num_chunks']} chunks · "
                        f"{round(doc['file_size_bytes']/1024, 1)} KB · {doc['file_type'].upper()}"
                    )
                with c2:
                    if st.button("🔁 Reindex", key=f"reindex_{doc['doc_id']}"):
                        with st.spinner("Reindexing…"):
                            try:
                                api_client.reindex_document(doc["doc_id"])
                                st.success("Reindexed!")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                with c3:
                    if st.button("🗑️ Delete", key=f"del_{doc['doc_id']}"):
                        with st.spinner("Deleting…"):
                            try:
                                api_client.delete_document(doc["doc_id"])
                                st.success("Deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
except Exception as e:
    st.error(f"Could not fetch documents: {e}")
