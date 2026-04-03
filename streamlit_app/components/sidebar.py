"""
streamlit_app/components/sidebar.py
"""
import streamlit as st
from streamlit_app.utils import api_client


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/books.png", width=64)
        st.title("RAG Assistant")
        st.caption("Production-grade document Q&A")
        st.divider()

        # API status
        try:
            api_client.health_check()
            st.success("✓ API online", icon="🟢")
            st.session_state["api_online"] = True
        except Exception:
            st.error("✗ API offline", icon="🔴")
            st.session_state["api_online"] = False

        st.divider()
        st.markdown("### Navigation")
        st.page_link("Home.py", label="🏠 Home")
        st.page_link("pages/1_Chat.py", label="💬 Chat")
        st.page_link("pages/2_Documents.py", label="📂 Documents")
        st.page_link("pages/3_Settings.py", label="⚙️ Settings")
        st.page_link("pages/4_Evaluation.py", label="📊 Evaluation")
