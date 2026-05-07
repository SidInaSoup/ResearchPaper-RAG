"""
Life Sciences Research Article Insight Extractor
─────────────────────────────────────────────────
Streamlit application for uploading life science PDFs,
building a FAISS index, and extracting structured insights via RAG.
"""

import json
import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from src.chunker import chunk_text
from src.embedder import embed_texts
from src.pdf_loader import extract_text_from_pdf
from src.rag_pipeline import TASKS, answer_query
from src.vector_store import build_index, index_exists, load_index, save_index
from src.llm_client import get_config

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

PAPERS_DIR = os.path.join("data", "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(os.path.join("data", "index"), exist_ok=True)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Life Sciences Insight Extractor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header Banner ──────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        color: #fff;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60%; left: -20%;
        width: 140%; height: 200%;
        background: radial-gradient(circle, rgba(0,255,180,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner h1 {
        margin: 0; font-size: 2rem; font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        margin: 0.5rem 0 0; opacity: 0.8; font-size: 1rem;
    }

    /* ── Cards ──────────────────────────────────────────── */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .info-card h4 {
        margin: 0 0 0.5rem;
        font-weight: 600;
        color: #4ecdc4;
    }

    /* ── Result Box ─────────────────────────────────────── */
    .result-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }

    /* ── Source Chunk ────────────────────────────────────── */
    .source-chunk {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #4ecdc4;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .source-chunk-header {
        font-weight: 600;
        color: #4ecdc4;
        margin-bottom: 0.35rem;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #4ecdc4;
    }

    /* ── Status Pills ───────────────────────────────────── */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(78,205,196,0.15);
        color: #4ecdc4;
        border: 1px solid rgba(78,205,196,0.3);
    }
    .status-empty {
        background: rgba(255,107,107,0.15);
        color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.3);
    }

    /* ── Tab Styling ────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <h1>🧬 Life Sciences Insight Extractor</h1>
    <p>Upload research papers, build a vector index, and extract structured insights using RAG.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📄 Upload Papers")

    uploaded_files = st.file_uploader(
        "Drop PDF research papers here",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        saved_count = 0
        for f in uploaded_files:
            dest = os.path.join(PAPERS_DIR, f.name)
            if not os.path.exists(dest):
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                saved_count += 1
        if saved_count:
            st.success(f"Saved {saved_count} new paper(s).")

    # Show papers on disk
    existing_papers = [
        p for p in os.listdir(PAPERS_DIR) if p.lower().endswith(".pdf")
    ]
    if existing_papers:
        st.markdown(f"**{len(existing_papers)} paper(s) on disk:**")
        for p in sorted(existing_papers):
            st.markdown(f"- `{p}`")
    else:
        st.info("No papers uploaded yet.")

    st.divider()

    # ── Index Controls ────────────────────────────────────────────────────────
    st.markdown("### 🗂️ Index")

    if index_exists():
        st.markdown('<span class="status-pill status-ready">● Index Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-empty">● No Index</span>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    build_btn = col1.button("Build Index", use_container_width=True, type="primary")
    rebuild_btn = col2.button("Rebuild", use_container_width=True)
    clear_btn = col3.button("Clear Data", use_container_width=True)

    if clear_btn:
        if os.path.exists(PAPERS_DIR):
            shutil.rmtree(PAPERS_DIR)
            os.makedirs(PAPERS_DIR)
        if os.path.exists(os.path.join("data", "index")):
            shutil.rmtree(os.path.join("data", "index"))
            os.makedirs(os.path.join("data", "index"))
        if "last_result" in st.session_state:
            del st.session_state["last_result"]
        st.success("All data and index cleared.")
        st.rerun()

    if build_btn or rebuild_btn:
        if not existing_papers:
            st.error("Upload at least one PDF first.")
        else:
            with st.spinner("Extracting text & building index…"):
                all_chunks = []
                progress = st.progress(0)
                for i, paper in enumerate(existing_papers):
                    pdf_path = os.path.join(PAPERS_DIR, paper)
                    pages = extract_text_from_pdf(pdf_path)
                    chunks = chunk_text(pages, paper)
                    all_chunks.extend(chunks)
                    progress.progress((i + 1) / len(existing_papers))

                if not all_chunks:
                    st.error("No text could be extracted from the uploaded PDFs.")
                else:
                    st.info(f"Generated {len(all_chunks)} chunks. Computing embeddings…")
                    texts = [c["text"] for c in all_chunks]
                    embeddings = embed_texts(texts)
                    index = build_index(embeddings)
                    save_index(index, all_chunks)
                    st.success(f"✅ Index built with {len(all_chunks)} chunks!")
                    st.rerun()

    st.divider()

    # ── Task Selection ────────────────────────────────────────────────────────
    st.markdown("### 🎯 Extraction Task")

    task_labels = {
        "custom": "🔍 Custom Question",
        "summarize": "📝 Summarize Paper",
        "background": "📚 Extract Background",
        "hypothesis": "💡 Extract Hypothesis",
        "methodology": "🔬 Extract Methodology",
        "results": "📊 Extract Key Results",
        "conclusion": "✅ Extract Conclusion",
        "limitations": "⚠️ Extract Limitations",
        "future_research": "🚀 Future Research Questions",
        "compare": "⚖️ Compare Papers",
    }

    selected_task = st.selectbox(
        "Choose a task",
        options=list(task_labels.keys()),
        format_func=lambda k: task_labels[k],
        key="task_select",
    )

    st.divider()

    # ── Retrieval Settings ────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")

    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=20, value=5, key="top_k")

    st.markdown("---")
    provider = get_config("LLM_PROVIDER", "openai")
    st.caption(f"LLM Provider: **{provider}**")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Query Input ───────────────────────────────────────────────────────────────

query_placeholder = (
    "e.g., What methods were used for protein expression analysis?"
    if selected_task == "custom"
    else f"Optional: refine the '{task_labels[selected_task]}' task with a specific question…"
)

query = st.text_area(
    "Ask a question about your papers",
    placeholder=query_placeholder,
    height=100,
    key="query_input",
)

run_btn = st.button("🚀 Run Extraction", type="primary", use_container_width=True)

# ── Execution ─────────────────────────────────────────────────────────────────

if run_btn:
    if not index_exists():
        st.error("⚠️ Please build the index first (sidebar → Build Index).")
    elif selected_task == "custom" and not query.strip():
        st.warning("Please enter a question for the custom task.")
    else:
        with st.spinner("Retrieving chunks & generating answer…"):
            try:
                result = answer_query(
                    query=query.strip() if query.strip() else TASKS.get(selected_task, ""),
                    task_type=selected_task,
                    top_k=top_k,
                )

                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

# ── Display Results ───────────────────────────────────────────────────────────

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    answer = result["answer"]
    source_chunks = result["source_chunks"]

    st.markdown("---")

    # Tabs: Markdown / JSON / Raw
    tab_md, tab_json, tab_raw = st.tabs(["📄 Markdown", "🧾 JSON", "🔧 Raw Response"])

    with tab_md:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        if "raw_response" in answer and len(answer) == 1:
            st.markdown(answer["raw_response"])
        else:
            if answer.get("paper_title"):
                st.markdown(f"## {answer['paper_title']}")

            fields = [
                ("background", "📚 Background"),
                ("hypothesis", "💡 Hypothesis"),
                ("methods", "🔬 Methods"),
                ("results", "📊 Results"),
                ("conclusion", "✅ Conclusion"),
                ("limitations", "⚠️ Limitations"),
                ("future_research_directions", "🚀 Future Research Directions"),
            ]

            for key, label in fields:
                value = answer.get(key)
                if not value:
                    continue
                st.markdown(f"### {label}")
                if isinstance(value, list):
                    for item in value:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(value)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_json:
        st.json(answer)

    with tab_raw:
        st.code(result.get("raw_response", ""), language="text")

    # ── Source Chunks ─────────────────────────────────────────────────────────
    if source_chunks:
        st.markdown("### 📎 Retrieved Source Chunks")

        for i, chunk in enumerate(source_chunks, 1):
            with st.expander(
                f"Chunk {i} — {chunk['filename']} (Page {chunk['page']}, Score: {chunk.get('score', 0):.4f})"
            ):
                st.markdown(
                    f'<div class="source-chunk">'
                    f'<div class="source-chunk-header">Chunk #{chunk["chunk_index"]} | {chunk["filename"]} | Page {chunk["page"]}</div>'
                    f'{chunk["text"][:1500]}{"…" if len(chunk["text"]) > 1500 else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Built with Streamlit · PyMuPDF · FAISS · Sentence-Transformers · OpenAI/Ollama"
)
