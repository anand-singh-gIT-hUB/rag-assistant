"""
streamlit_app/components/chat_box.py
"""
import streamlit as st


def render_chat_history(messages: list[dict]):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_citations(citations: list[dict]):
    if not citations:
        return
    with st.expander(f"📎 {len(citations)} source(s) cited", expanded=False):
        for i, c in enumerate(citations, 1):
            page_str = f" · page {c['page_number']}" if c.get("page_number") else ""
            section_str = f" · {c['section_title']}" if c.get("section_title") else ""
            st.markdown(
                f"**[{i}] {c['file_name']}**{page_str}{section_str}\n\n"
                f"> {c['excerpt']}"
            )
            if i < len(citations):
                st.divider()
