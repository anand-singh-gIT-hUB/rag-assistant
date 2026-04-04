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
        try:
            stream = api_client.stream_query(
                question=question,
                top_k=st.session_state.get("top_k"),
                rerank=st.session_state.get("reranker_enabled"),
                doc_ids=st.session_state.get("selected_doc_ids") or None,
            )
            
            answer_placeholder = st.empty()
            answer_placeholder.markdown("*(Retrieving documents...)*")
            
            full_answer = ""
            citations = []
            metrics = {}
            
            for chunk in stream:
                if chunk["type"] == "metadata":
                    citations = chunk.get("citations", [])
                    metrics["model"] = chunk.get("model", "")
                    metrics["retrieval_top_k"] = chunk.get("retrieval_top_k", 0)
                    metrics["reranked"] = chunk.get("reranked", False)
                    metrics["retrieval_time"] = chunk.get("retrieval_time", 0)
                elif chunk["type"] == "token":
                    full_answer += chunk["content"]
                    answer_placeholder.markdown(full_answer + "▌")
                elif chunk["type"] == "error":
                    st.error(chunk["detail"])
                    break
                elif chunk["type"] == "done":
                    metrics["gen_time"] = chunk.get("gen_time", 0)
                    metrics["total_time"] = chunk.get("total_time", 0)
                    break

            answer_placeholder.markdown(full_answer)
            st.session_state["messages"].append({"role": "assistant", "content": full_answer})
            st.session_state["citations"] = citations

            render_citations(citations)

            with st.expander("ℹ️ Query info"):
                st.write(f"**Model:** {metrics.get('model', 'N/A')}")
                st.write(f"**Retrieval top-k:** {metrics.get('retrieval_top_k', 0)}")
                st.write(f"**Reranked:** {metrics.get('reranked', False)}")
                st.write(f"**Retrieval Time:** {metrics.get('retrieval_time', 0):.2f}s")
                st.write(f"**Generation Time:** {metrics.get('gen_time', 0):.2f}s")
                st.write(f"**Total Time:** {metrics.get('total_time', 0):.2f}s")

        except Exception as e:
            err = f"Error: {e}"
            st.error(err)
            st.session_state["messages"].append({"role": "assistant", "content": err})
