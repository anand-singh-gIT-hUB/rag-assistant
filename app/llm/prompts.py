"""
app/llm/prompts.py
───────────────────
Prompt templates for the RAG system.
"""

SYSTEM_PROMPT = """\
You are a precise research assistant. 
Use ONLY the context below to answer. 
- If context is insufficient, say: "I don't have enough information."
- Use cite [1], [2] to support claims.
- Be extremely concise. Use bullet points.
"""


def build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the user message with numbered context passages."""
    passages = []
    for i, chunk in enumerate(context_chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("file_name", "unknown")
        page = meta.get("page_number", "")
        page_str = f", page {page}" if page and page != -1 else ""
        passages.append(f"[{i}] (source: {source}{page_str})\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(passages)
    return f"Context passages:\n\n{context_block}\n\nQuestion: {question}"
