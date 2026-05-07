"""
RAG pipeline: retrieve relevant chunks and generate structured answers.
"""

import json

from .embedder import embed_texts
from .llm_client import generate, parse_json_response
from .vector_store import load_index, search

# ── Predefined Tasks ──────────────────────────────────────────────────────────

TASKS = {
    "summarize": "Summarize the paper(s) concisely.",
    "background": "Extract the background and motivation of the research.",
    "hypothesis": "Extract the hypothesis or research question.",
    "methodology": "Extract the methodology and experimental design used.",
    "results": "Extract the key results and findings.",
    "conclusion": "Extract the conclusion of the paper(s).",
    "limitations": "Extract the limitations mentioned by the authors.",
    "future_research": "Generate future research questions based on the paper(s).",
    "compare": "Compare the uploaded papers and highlight differences and similarities.",
    "custom": "",  # user provides their own question
}

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a life sciences research assistant.
Use only the provided context from research papers.
Extract accurate information and avoid hallucination.
If information is not present in the context, say "Not found in provided papers."
Always return your answer as valid JSON matching the requested schema."""

# ── JSON Output Schema ────────────────────────────────────────────────────────

OUTPUT_SCHEMA = """\
{
  "paper_title": "",
  "background": "",
  "hypothesis": "",
  "methods": [],
  "results": [],
  "conclusion": "",
  "limitations": "",
  "future_research_directions": [],
  "source_chunks": []
}"""

# ── User Prompt Template ─────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
Task:
{task_description}

User question:
{query}

Context (retrieved from research papers):
{context}

Return the answer in this JSON format (fill in only the fields relevant to the task, leave others as empty strings or empty lists):
{schema}
"""


def _build_context_string(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Chunk {i} | {chunk['filename']} | Page {chunk['page']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer_query(
    query: str,
    task_type: str = "custom",
    top_k: int = 5,
) -> dict:
    """
    Full RAG pipeline: embed query → retrieve chunks → call LLM → parse response.

    Args:
        query: The user's question or the predefined task key.
        task_type: One of the TASKS keys.
        top_k: Number of chunks to retrieve.

    Returns:
        Dict with keys:
            - answer: parsed JSON response from the LLM
            - source_chunks: list of retrieved chunk metadata
            - raw_response: raw LLM text (for debugging)
    """
    # 1. Determine the task description
    task_description = TASKS.get(task_type, "")
    if task_type == "custom" and not query.strip():
        return {
            "answer": {"error": "Please enter a question."},
            "source_chunks": [],
            "raw_response": "",
        }

    effective_query = query if query.strip() else task_description

    # 2. Embed the query
    query_embedding = embed_texts([effective_query])

    # 3. Retrieve relevant chunks
    index, metadata = load_index()
    retrieved_chunks = search(index, query_embedding, metadata, top_k=top_k)

    # 4. Build the prompt
    context_str = _build_context_string(retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(
        task_description=task_description,
        query=effective_query,
        context=context_str,
        schema=OUTPUT_SCHEMA,
    )

    # 5. Call the LLM
    raw_response = generate(prompt, system_prompt=SYSTEM_PROMPT)

    # 6. Parse the response
    answer = parse_json_response(raw_response)

    # Inject source chunk references
    answer["source_chunks"] = [
        {
            "filename": c["filename"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "score": c.get("score", 0),
        }
        for c in retrieved_chunks
    ]

    return {
        "answer": answer,
        "source_chunks": retrieved_chunks,
        "raw_response": raw_response,
    }
