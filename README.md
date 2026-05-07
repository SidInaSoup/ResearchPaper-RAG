# 🧬 Life Sciences Research Article Insight Extractor

A **RAG-based** Streamlit application that lets you upload life sciences research paper PDFs, index them with FAISS, and extract structured insights like background, hypothesis, methods, results, and conclusions using an LLM.

---

## Features

- **PDF Upload** — Upload one or more research paper PDFs via the Streamlit sidebar.
- **Automatic Text Extraction** — Uses PyMuPDF to extract page-level text from each PDF.
- **Vector Indexing** — Chunks text into ~1000-word segments, embeds them with `all-MiniLM-L6-v2`, and stores them in a FAISS index.
- **Predefined Extraction Tasks** — One-click extraction for background, hypothesis, methodology, results, conclusions, limitations, future research, and paper comparison.
- **Custom Questions** — Ask free-form questions about your papers.
- **Structured Output** — Results are returned as structured JSON and formatted Markdown.
- **Source Chunk Attribution** — Every answer shows the retrieved source chunks used to generate it.
- **LLM Flexibility** — Supports OpenAI API and Ollama local models (switchable via env var).

---

## Project Structure

```
life_science_rag/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md
├── data/
│   ├── papers/             # Uploaded PDFs
│   └── index/              # FAISS index + metadata
└── src/
    ├── __init__.py
    ├── pdf_loader.py       # PDF text extraction
    ├── chunker.py          # Text chunking with overlap
    ├── embedder.py         # Sentence-transformer embeddings
    ├── vector_store.py     # FAISS build / save / load / search
    ├── rag_pipeline.py     # RAG query pipeline
    └── llm_client.py       # OpenAI / Ollama LLM client
```

---

## Setup

### 1. Clone & Install

```bash
cd life_science_rag
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure LLM

Copy the example env file and edit it:

```bash
cp .env.example .env
```
### Create a .env file with the below template

```env
# ── LLM Provider ──────────────────────────────────────────
# "openai" or "ollama"
LLM_PROVIDER=openai

# ── OpenAI ────────────────────────────────────────────────
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini

# ── Ollama (optional) ────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

#### Using OpenAI

Set your API key in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

#### Using Ollama (local)

1. Install [Ollama](https://ollama.ai/) and pull a model:
   ```bash
   ollama pull llama3
   ```
2. Update `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3
   ```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

### Upload Papers

1. Open the **sidebar** (left panel).
2. Click **Upload PDFs** and select 1–5 research paper PDFs.
3. Papers are stored in `data/papers/`.

### Build the Index

1. Click **Build Index** in the sidebar.
2. The app extracts text, chunks it, generates embeddings, and saves the FAISS index.
3. Status changes to **● Index Ready**.

### Extract Insights

1. Choose a **task** from the sidebar dropdown (e.g., *Extract Methodology*).
2. Optionally type a **custom question** in the main area.
3. Click **🚀 Run Extraction**.
4. View results in **Markdown**, **JSON**, or **Raw** tabs.
5. Expand **Source Chunks** below to see which parts of the papers were used.

---

## Example Questions

| Task | Example Question |
|------|-----------------|
| Custom | *What gene editing technique was used and why?* |
| Custom | *What statistical tests were applied to the data?* |
| Custom | *How large was the sample size?* |
| Summarize | *(leave blank — auto-summarizes)* |
| Compare | *Compare the methodologies across the uploaded papers.* |

---

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **PyMuPDF** — PDF text extraction
- **FAISS** — Vector similarity search
- **sentence-transformers** — Text embeddings (`all-MiniLM-L6-v2`)
- **OpenAI API / Ollama** — LLM generation

---

## License

MIT
