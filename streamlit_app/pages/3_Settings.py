"""
streamlit_app/pages/3_Settings.py
───────────────────────────────────
Runtime settings stored in session state.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Settings — RAG Assistant", page_icon="⚙️", layout="wide")

from streamlit_app.utils.session import init_session
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.utils import api_client

init_session()
render_sidebar()

st.title("⚙️ Settings")
st.caption("These settings are applied to new queries in the current session.")

st.subheader("Retrieval")
st.session_state["top_k"] = st.slider(
    "Top-K chunks to retrieve",
    min_value=1, max_value=50,
    value=st.session_state.get("top_k", 10),
    help="Number of chunks retrieved before reranking.",
)
st.session_state["reranker_enabled"] = st.toggle(
    "Enable cross-encoder reranker",
    value=st.session_state.get("reranker_enabled", True),
    help="Reranker improves precision but adds ~200ms latency.",
)

st.divider()
st.subheader("Document Filtering")
try:
    data = api_client.list_documents()
    docs = data.get("documents", [])
    options = {d["file_name"]: d["doc_id"] for d in docs}
    selected_names = st.multiselect(
        "Limit queries to specific documents (empty = all)",
        options=list(options.keys()),
    )
    st.session_state["selected_doc_ids"] = [options[n] for n in selected_names]
except Exception:
    st.warning("Could not load document list (API offline?).")

st.divider()
st.subheader("API Status")
try:
    health = api_client.health_check()
    st.json(health)
except Exception as e:
    st.error(f"API unreachable: {e}")
