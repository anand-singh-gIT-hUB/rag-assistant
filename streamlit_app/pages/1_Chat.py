"""
streamlit_app/pages/1_Chat.py
──────────────────────────────
Interactive chat page with citations.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Chat — RAG Assistant", page_icon="💬", layout="wide")

from streamlit_app.utils.session import init_session, reset_chat
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.chat_box import render_chat_history, render_citations
from streamlit_app.utils import api_client

init_session()
render_sidebar()

st.title("💬 Chat")

# ── Top controls ──────────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("🗑️ Clear chat", use_container_width=True):
        reset_chat()

# ── Chat history ──────────────────────────────────────────────────────────────
render_chat_history(st.session_state["messages"])

# ── Latest citations ──────────────────────────────────────────────────────────
render_citations(st.session_state.get("citations", []))

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.get("api_online"):
        st.error("API is offline. Please start the backend server.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = api_client.query(
                    question=question,
                    top_k=st.session_state.get("top_k", 10),
                    rerank=st.session_state.get("reranker_enabled", True),
                    doc_ids=st.session_state.get("selected_doc_ids") or None,
                )
                answer = response["answer"]
                citations = response.get("citations", [])

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                st.session_state["citations"] = citations

                render_citations(citations)

                with st.expander("ℹ️ Query info"):
                    st.write(f"**Model:** {response['model']}")
                    st.write(f"**Chunks used:** {response['retrieval_top_k']}")
                    st.write(f"**Reranked:** {response['reranked']}")
            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err})
