"""
streamlit_app/pages/3_Settings.py
───────────────────────────────────
Runtime settings stored in session state.
Settings are optional overrides — when left at "Use backend default",
the backend .env values are used. Only explicit changes are sent.
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
st.caption("These settings are **optional overrides** for the current session. Leave them off to use the backend defaults from `.env`.")

# ── Retrieval Overrides ────────────────────────────────────────────────────────
st.subheader("Retrieval Overrides")

override_top_k = st.toggle(
    "Override retrieval Top-K",
    value=st.session_state.get("top_k") is not None,
    help="Enable to manually set how many chunks are retrieved. Default is controlled by RETRIEVAL_TOP_K in .env.",
)
if override_top_k:
    st.session_state["top_k"] = st.slider(
        "Top-K chunks to retrieve",
        min_value=1, max_value=50,
        value=st.session_state.get("top_k") or 5,
        help="More chunks = more context but slower generation.",
    )
else:
    st.session_state["top_k"] = None
    st.caption("Using backend default (RETRIEVAL_TOP_K in .env)")

override_reranker = st.toggle(
    "Override reranker setting",
    value=st.session_state.get("reranker_enabled") is not None,
    help="Enable to manually control reranking. Default is controlled by RERANKER_ENABLED in .env.",
)
if override_reranker:
    st.session_state["reranker_enabled"] = st.toggle(
        "Enable cross-encoder reranker",
        value=st.session_state.get("reranker_enabled") or False,
        help="Reranker improves precision but adds significant extra latency on CPU.",
    )
else:
    st.session_state["reranker_enabled"] = None
    st.caption("Using backend default (RERANKER_ENABLED in .env)")

st.divider()

# ── Document Filtering ────────────────────────────────────────────────────────
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

# ── API Status ─────────────────────────────────────────────────────────────────
st.subheader("API Status")
try:
    health = api_client.health_check()
    st.json(health)
except Exception as e:
    st.error(f"API unreachable: {e}")
