"""
app/llm/prompts.py
───────────────────
Prompt templates for the RAG system.
Designed for speed on small local models:
- Compressed context (max ~400 chars per chunk)
- Adaptive answer instructions based on question intent
- Structured output format enforced in the prompt
"""

SYSTEM_PROMPT = """\
You are a document Q&A assistant.
Answer only from the provided context.
If context is insufficient, say: "I don't have enough information."
Be specific and structured.\
"""

# ── Context Compression ───────────────────────────────────────────────────────

def compress_context(text: str, max_chars: int = 400) -> str:
    """Normalize whitespace and truncate to max_chars."""
    text = " ".join(text.split())
    return text[:max_chars]


# ── Answer Style ──────────────────────────────────────────────────────────────

def build_answer_instruction(question: str) -> str:
    """
    Return a concise or detailed instruction based on question intent.
    Detailed mode is triggered by keywords like 'explain', 'more detail', etc.
    """
    q = question.lower()
    if any(kw in q for kw in ("more detail", "explain", "elaborate", "describe", "how does", "why")):
        return (
            "Provide a detailed answer in 5-7 short bullet points. "
            "Be specific, avoid filler."
        )
    return (
        "Provide a concise structured answer in 3 short sections. "
        "Keep it under 120 words."
    )


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build an optimised user message:
    - Each chunk is compressed to ~400 chars
    - Answer format instruction is inferred from the question
    - Minimal separator (newline) instead of verbose '---' blocks
    """
    context_blocks = []
    for chunk in context_chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("file_name", "unknown")
        page = meta.get("page_number", "")
        page_str = f" p{page}" if page and page != -1 else ""
        short_text = compress_context(chunk.get("text", ""), max_chars=400)
        context_blocks.append(f"[Source: {source}{page_str}]\n{short_text}")

    context = "\n\n".join(context_blocks)
    answer_instruction = build_answer_instruction(question)

    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer format: {answer_instruction}"
    )
