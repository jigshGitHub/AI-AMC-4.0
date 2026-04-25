# RealEstate (RAG) — README

This folder contains a small Retrieval-Augmented Generation (RAG) example for real estate documents.
It uses a local Chroma vector store for retrieval, a local HuggingFace embedding model, and an LLM for planning and final answer generation through a LangGraph-style state machine.

Files of interest
- `agent.py` — the LangGraph agent: defines the shared state, retrieval + multi-specialist analysis nodes, planning, and final answer composition.
- `ingestion.py` — ingestion pipeline: loads PDFs, splits into overlapping chunks, computes embeddings, and writes a Chroma DB to disk.
- `run.py` — small CLI wrapper: checks for vector DB, runs the interactive chat loop, calls `agent.query_rag` for answers.
- `config.py` — configuration via environment variables (see 'Environment' below).

Quick summary of how it works

1. Ingestion: run `ingestion.py` to read PDFs from the `Data/` folder, split pages into chunks, embed them, and persist a Chroma DB.
2. Query: run `run.py` to start a prompt loop. For each user question it:
   - builds a LangGraph app (from `agent.py`)
   - understands the question (analysis)
   - searches the Chroma index for relevant chunks
   - runs three "specialists" in parallel (market, property, investment)
   - runs a planner to choose quick vs. detailed answer
   - generates the final answer using the selected LLM prompt template

Data flow (high level)

User -> `run.py` -> `agent.query_rag()`
    -> LangGraph START
      -> `understand_question` (question_analysis)
      -> `search_index` (retrieved_documents, retrieved_context, retrieved_sources)
         -> parallel: `market_specialist`, `property_insights_specialist`, `investment_strategy_specialist`
         -> `pick_response_mode` (decides quick vs. detailed)
         -> `quick_answer` OR `detailed_answer`
      -> END (final_answer returned)

Detailed analysis

agent.py
- Purpose: encapsulates the complete RAG agent and LangGraph-style pipeline.
- Key elements:
  - `RealEstateState` (Pydantic BaseModel): typed container for the shared state that flows between nodes. Important fields: `user_question`, `retrieved_documents`, `retrieved_context`, `market_analysis`, `property_insights`, `investment_strategy`, `needs_detailed_answer`, `final_answer`, and `messages`.
  - `understand_question(state)`: uses the LLM to summarize/interpret the user's intent before retrieval.
  - `search_index(state)`: loads the Chroma DB (via `load_vector_store()`), calls `similarity_search(...)` to fetch top-K chunks, then formats context and sources.
  - `market_specialist`, `property_insights_specialist`, `investment_strategy_specialist`: three parallel LLM calls that analyze the retrieved context from different angles and write results back into the shared state.
  - `pick_response_mode(state)`: uses the LLM planner to decide whether a quick or detailed answer is appropriate. It expects strict JSON from the model and falls back to a quick answer when parsing fails.
  - `quick_answer` / `detailed_answer`: generate the final LLM output from the assembled context. `detailed_answer` includes structured sections; `quick_answer` returns a concise paragraph or bullets.
  - Utility functions: `format_context`, `format_sources`, `build_embedding_model`, `load_vector_store`, `query_rag`.

Notes / potential gotchas in `agent.py`:
- `pick_response_mode` expects the planner LLM to emit JSON exactly as shown; if the LLM returns non-JSON text the code falls back to quick mode — consider adding stronger validation and a retry with a stricter system prompt.
- `load_vector_store()` raises FileNotFoundError when the Chroma DB path is missing — that's intentional to force users to run ingestion.
- LLM calls use `llm.invoke(...).content` — depending on your LangChain/OpenAI wrapper this may need to be adapted (some libs use `.text` or `.content[0].text`). Keep an eye on API compatibility.

ingestion.py
- Purpose: convert PDF pages into embedding-ready chunks and persist them to Chroma.
- Key steps:
  - `get_pdf_paths(data_dir)`: enumerates PDF files in `Data/`.
  - `load_pdf_documents(pdf_paths)`: loads each PDF page using `PyPDFLoader` into a list of `Document` objects (one per page).
  - `split_documents(documents)`: uses `RecursiveCharacterTextSplitter` with parameters from `config.py` to create overlapping chunks.
  - `create_vector_store(chunks)`: computes embeddings using `HuggingFaceEmbeddings` and writes a Chroma DB at `CHROMA_DB_DIR`.

Notes / potential gotchas in `ingestion.py`:
- `PyPDFLoader` extracts text but does not OCR images — scanned/picture PDFs may produce empty text. Use OCR (e.g., Tesseract) if you need scanned PDF support.
- Chunking parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`) live in `config.py`; tweak them depending on document style.
- The text splitter uses separators that include an empty string — this matches the original implementation but may behave unexpectedly in rare cases; test on your corpus.

run.py
- Purpose: a small CLI entrypoint for interactive question-answering.
- Behavior:
  - Prints a banner and verifies whether the vector DB exists (via `config.CHROMA_DB_DIR`).
  - If DB is missing, prints setup instructions (put PDFs in `Data/`, run `ingestion.py`).
  - If DB exists, starts a REPL loop that reads user questions and calls `agent.query_rag` for answers.

Environment (variables from `config.py`)
- OPENAI_API_KEY: (string) OpenAI key used by the LLM wrapper.
- DATA_DIR: (string) path to the folder with PDFs. Example: `./Data`.
- CHROMA_DB_DIR: (string) path where Chroma persists vectors. Example: `./Chroma_DB`.
- EMBEDDING_MODEL: (string) HuggingFace model name for embeddings. Example: `sentence-transformers/all-MiniLM-L6-v2`.
- LLM_MODEL: (string) model identifier used by `ChatOpenAI` wrapper in this repo.
- DEBUG: (bool) enable debug logging.
- TOP_K: (int) number of chunks to retrieve (defaults to 4).
- TEMPERATURE: (float) sampling temperature for LLM; a value between 0.0 and 1.0. `config.py` now parses this as a float and defaults to `0.0` for deterministic outputs.
- CHUNK_SIZE / CHUNK_OVERLAP: ints controlling chunking (defaults in `config.py`: 1000 / 200).

Quick start (PowerShell)

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

This project includes a `requirements.txt` inside `RAG/RealEstate` with the likely dependencies used by the example. Use it to install pinned packages for the example.

```powershell
pip install -r RAG\RealEstate\requirements.txt
```

3. Configure environment variables

Create a `.env` at the project root (or set environment variables) and set at minimum:

```
OPENAI_API_KEY=sk-...
DATA_DIR=./Data
CHROMA_DB_DIR=./Chroma_DB
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-4o-mini  # or your preferred model name
TOP_K=4
TEMPERATURE=0
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

4. Put your PDFs into `RAG/RealEstate/Data/` and run ingestion

```powershell
cd RAG\RealEstate
python ingestion.py
```

5. Run the chat app

```powershell
python run.py
```

Troubleshooting & tips
- "Vector database not found" — ensure `CHROMA_DB_DIR` points at the folder created by `ingestion.py` and contains Chroma files. The ingestion script logs where it saves the DB.
- Empty retrieval results — check that the embedding model is compatible and that your PDFs contain selectable text (not scanned images).
- Planner JSON parse errors — the planner expects strict JSON output; noisy LLM completions will force fallback to a quick answer. Consider constraining the model temperature (0) and adding a JSON schema enforcement step.
- Tests — basic unit tests for `agent.pick_response_mode` were added under `RAG/RealEstate/tests/`. These use `pytest` and monkeypatch the `llm.invoke` method to simulate clean/noisy responses. To run them:

```powershell
cd RAG\RealEstate
pytest -q
```
- Embedding / LLM API mismatches — the code calls `.invoke(...).content` on LLM responses; if your LLM wrapper uses `.text` or returns a different structure, adapt the calls in `agent.py`.
- Scanned PDFs — use OCR (Tesseract or commercial APIs) to extract text before ingestion.

Next improvements you can make (low-risk steps)
- Add a `requirements.txt` or `pyproject.toml` pinned to tested versions.
- Add CLI args to `run.py` for `--data-dir` and `--db-dir`.
- Add retries and `429` handling to LLM calls.
- Add a small test that runs ingestion on a tiny PDF and asserts the Chroma DB is created.
- Change `TEMPERATURE` parsing in `config.py` to float: `float(os.getenv("TEMPERATURE", 0.0))`.

If you want, I can:
- create a `requirements.txt` with the likely packages and pinned versions,
- add a small unit test for `ingestion.run_ingestion()` (using a tiny sample PDF), or
- tighten `agent.py`'s planner JSON parsing with retries and schema validation.

---
Generated on: 2026-04-24
