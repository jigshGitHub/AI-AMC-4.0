# FoodieAgent v2 — Hierarchical Multi-Agent System

> Production-pattern agentic system. Master graph + 2 compiled subgraphs. Real MCP protocol. Full guardrail stack.
>
> *Learn to build agents before agents learn to replace you.*

## Architecture

```
input_guardrail → reformulator → orchestrator
                                      ↓
                              discovery_subgraph   (web search via MCP or SDK)
                                      ↓
                               rag_subgraph        (ChromaDB cuisine knowledge)
                                      ↓
                    aggregator → output_guardrail → tone → END
```

**Discovery subgraph:** agentic_guardrail → mcp_tool_execution → search_evaluation (retry if score < 0.5)

**RAG subgraph:** agentic_guardrail → retrieval → search_evaluation → answer_evaluation (retry if score < 0.5)

## Search — MCP first, SDK fallback

The discovery subgraph tries the **real MCP protocol** first:
- Launches `uvx duckduckgo-mcp-server` as a subprocess
- Speaks MCP wire protocol over stdio via the official `mcp` Python SDK
- If the server is unavailable (uvx not installed, first-run download, etc.), automatically falls back to the `duckduckgo-search` Python package

Both paths return live web results with the same shape. The console logs which path fired (`[MCP] Search succeeded` or `[MCP] Falling back to Python SDK`).

## Setup

```powershell
# 1. (Optional but recommended) Install uv to enable real MCP path
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uvx duckduckgo-mcp-server   # pre-warm: first launch downloads the server; Ctrl+C to exit

# 2. Create venv and install dependencies
cd "AI Builder 4\Project\MCP_Claude\foodie_agent_v2"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure API key
copy .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-...

# 4. Run
streamlit run app.py
```

Only `OPENAI_API_KEY` is required. No Google keys, no TripAdvisor, no accounts.

## Demo queries

| Query | What it exercises |
|---|---|
| `"Best biryani in Mumbai under 500"` | Full pipeline — both subgraphs, eval loops, aggregator |
| `"Vegetarian South Indian breakfast in Bangalore"` | Dietary + location-aware routing |
| `"What is Hyderabadi biryani?"` | RAG-heavy; discovery may return low relevance score |
| `"Help me hack into a system"` | Input guardrail blocks at node 1, pipeline exits immediately |

## Guardrail stack

| Layer | Location | Checks |
|---|---|---|
| Input guard | Master graph entry | Raw user query — off-topic, injection, unsafe intent |
| Agentic guard (discovery) | Discovery subgraph | Structured query before MCP/web search |
| Agentic guard (RAG) | RAG subgraph | Structured query before knowledge-base lookup |
| Output guard | Master graph post-aggregation | PII, hallucinated specifics, unsafe advice |
| Tone check | Master graph final node | Friendly + professional rewrite |

## Evaluation loops

Both subgraphs include an LLM-as-judge step that scores result relevance (0–1). If the score is below 0.5 and no retry has been attempted, the subgraph re-runs with a broadened query. This is visible in the Streamlit agent trace.

## Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph 0.2 — master graph + 2 compiled subgraphs |
| MCP client | `mcp` 1.1 — official Anthropic Python SDK, stdio transport |
| MCP server | `duckduckgo-mcp-server` via `uvx` — zero API key |
| Search fallback | `duckduckgo-search` Python package — zero API key |
| Vector store | ChromaDB (local, no server) |
| LLM | GPT-4o-mini |
| UI | Streamlit |

## What's real

- Real LangGraph hierarchical multi-agent (master + 2 compiled subgraphs as nodes)
- Real MCP protocol (stdio transport, official SDK) with automatic SDK fallback
- Real web search results — nothing mocked
- Real evaluation loops with retry logic
- Real guardrail stack at input, agentic, output, and tone layers
