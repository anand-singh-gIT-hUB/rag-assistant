"""
streamlit_app/utils/session.py
────────────────────────────────
Streamlit session-state helpers.
"""
import streamlit as st


def init_session():
    defaults = {
        "messages": [],          # Chat history: list of {"role": str, "content": str}
        "citations": [],          # Latest citations
        "selected_doc_ids": [],   # Filtered doc IDs
        "reranker_enabled": None,
        "top_k": None,
        "api_online": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_chat():
    st.session_state["messages"] = []
    st.session_state["citations"] = []
