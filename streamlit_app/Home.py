"""
streamlit_app/Home.py
──────────────────────
Landing page for the RAG Knowledge Assistant.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_app.utils.session import init_session
from streamlit_app.components.sidebar import render_sidebar

init_session()
render_sidebar()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size:3rem; font-weight:800; margin-bottom:0.2em;">
            🧠 RAG Knowledge Assistant
        </h1>
        <p style="font-size:1.2rem; color:#888; max-width:600px; margin:auto;">
            Upload your documents. Ask anything. Get grounded, cited answers — powered
            by OpenAI or a fully local open-source stack.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Feature cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 📂 Ingest")
    st.markdown("Upload **PDF, DOCX, TXT, Markdown** files. They are parsed, chunked, and stored in ChromaDB automatically.")
with col2:
    st.markdown("### 💬 Chat")
    st.markdown("Ask any question. The assistant retrieves the most relevant passages and produces a grounded, cited answer.")
with col3:
    st.markdown("### 🔁 Rerank")
    st.markdown("A cross-encoder reranker sharpens retrieval precision before the LLM sees any context.")
with col4:
    st.markdown("### 📊 Evaluate")
    st.markdown("Run **Ragas** benchmarks — faithfulness, answer relevancy, context recall & precision — in one click.")

st.divider()
st.markdown(
    """
    **Quick start:**
    1. Go to **📂 Documents** and upload a file.
    2. Open **💬 Chat** and ask a question.
    3. Review citations and tweak settings in **⚙️ Settings**.
    4. Benchmark quality in **📊 Evaluation**.
    """,
)
