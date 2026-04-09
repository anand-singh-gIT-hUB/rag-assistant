"""
app/llm/prompts.py
───────────────────
Prompt templates for the RAG system.
Designed for maximum Answer Relevancy on local small models:
- Heavily compressed context chunks (max 350 chars).
- Strict, direct-first structure.
"""

SYSTEM_PROMPT = """\
You are a precise document question-answering assistant.

Answer the user's question directly in the first sentence.
Then provide 2 to 4 concise supporting details from the retrieved context.

Rules:
- Use only the provided context.
- Do not add generic background information unless it directly helps answer the question.
- If the context is insufficient, say exactly what is missing.
- Be specific and relevant.
- Keep the answer compact but informative.\
"""

# ── Context Compression ───────────────────────────────────────────────────────

def compress_context(text: str, max_chars: int = 350) -> str:
    """Normalize whitespace and truncate to max_chars to reduce noise."""
    text = " ".join(text.split())
    return text[:max_chars]


# ── Answer Style ──────────────────────────────────────────────────────────────

def build_answer_instruction(question: str) -> str:
    q = question.lower()

    if q.startswith("what is") or q.startswith("what are"):
        return "Give a direct definition first, then 2 to 3 specific details."
    if q.startswith("how"):
        return "Explain the process directly and clearly in 3 to 4 short points."
    if q.startswith("why"):
        return "State the main reason first, then give concise supporting points."
    return "Answer directly, then give a few specific supporting details."


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build an optimised user message tailored to local models:
    - Instructs clearly with 'Task:' block.
    """
    context_blocks = []
    for chunk in context_chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("file_name", "unknown")
        short_text = compress_context(chunk.get("text", ""), max_chars=350)
        context_blocks.append(f"[Source: {source}]\n{short_text}")

    context = "\n\n".join(context_blocks)
    answer_instruction = build_answer_instruction(question)

    return f"""
Question:
{question}

Context:
{context}

Task:
Answer the question directly using only the context.
{answer_instruction}
"""
