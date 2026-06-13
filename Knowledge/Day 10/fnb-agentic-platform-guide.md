# F&B Agentic AI Platform — Complete Blueprint

## Cognithic AI Labs | Agentic AI Builder Expert — Batch 4.0

---

## Part 1: Folder Structure

```
fnb-agentic-platform/
│
├── README.md                          # Project overview, setup, architecture
├── CONTRIBUTING.md                    # Git workflow, MR rules for students
├── STUDENT_GUIDE.md                   # Step-by-step onboarding guide
├── .gitignore
├── .env.example                       # Template — never commit real .env
├── requirements.txt                   # Root dependencies (shared)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── main.py                            # Entry point — uvicorn launcher
│
├── config/
│   ├── __init__.py
│   ├── settings.py                    # Pydantic Settings (env vars)
│   └── constants.py                   # System-wide constants
│
├── core/                              # ★ SHARED FRAMEWORK — DO NOT MODIFY
│   ├── __init__.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── main_graph.py              # LangGraph main StateGraph
│   │   ├── state.py                   # AgentState TypedDict
│   │   ├── router.py                  # Intent → SubAgent routing
│   │   └── execution_strategy.py      # Parallel vs sequential tool dispatch
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── regex_guardrail.py         # Regex-based input/output filters
│   │   ├── agent_guardrail.py         # LLM-based guardrail agent
│   │   ├── pre_tool_guardrail.py      # Check before tool/subgraph execution
│   │   └── output_guardrail.py        # Final response guardrail
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   └── reformulator.py            # Query reformation node
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── search_service.py          # ChromaDB vector search
│   │   ├── search_evaluator.py        # Evaluate retrieval quality
│   │   └── embeddings.py              # Embedding model wrapper
│   │
│   ├── response/
│   │   ├── __init__.py
│   │   ├── aggregator.py              # Merge multi-tool responses
│   │   ├── answer_evaluator.py        # Score answer quality
│   │   ├── tone_checker.py            # Brand voice enforcement
│   │   └── citation_builder.py        # Source attribution + timing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_state.py              # Shared state schemas (Pydantic)
│   │   ├── tool_schemas.py            # Base tool I/O schemas
│   │   └── api_schemas.py             # Request/Response models
│   │
│   └── utils/
│       ├── __init__.py
│       ├── llm_factory.py             # LLM instantiation helper
│       ├── logger.py                  # Structured logging (structlog)
│       └── timing.py                  # Execution timer decorator
│
├── subagents/                         # ★ EACH STUDENT WORKS HERE
│   ├── __init__.py
│   ├── registry.py                    # Auto-discovers & registers subagents
│   │
│   ├── product_ideation/              # ── SubAgent 1: Product Ideation
│   │   ├── __init__.py
│   │   ├── README.md                  # SubAgent-specific docs
│   │   ├── agent.py                   # LangGraph subgraph definition
│   │   ├── state.py                   # SubAgent-specific state
│   │   ├── prompts.py                 # System prompts
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── trend_researcher.py    # Web search for F&B trends
│   │   │   ├── concept_generator.py   # LLM-powered ideation
│   │   │   └── competitor_analyzer.py # Market analysis
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── models.py             # Pydantic models for this agent
│   │   ├── data/
│   │   │   └── seed.json              # Sample data for testing
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_agent.py
│   │
│   ├── product_catalog/               # ── SubAgent 2: Product Catalog
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── catalog_search.py      # RAG search over product DB
│   │   │   ├── catalog_crud.py        # Add/update/delete products
│   │   │   ├── pairing_engine.py      # AI food pairings
│   │   │   └── bundle_creator.py      # Combo meal generation
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_agent.py
│   │
│   ├── customer_loyalty/              # ── SubAgent 3: Customer Loyalty
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── points_engine.py       # Earn/burn/balance
│   │   │   ├── tier_manager.py        # Tier evaluation/upgrade
│   │   │   ├── wallet_manager.py      # Digital wallet ops
│   │   │   └── offer_engine.py        # Promos, missions, coupons
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_agent.py
│   │
│   ├── cross_sell_upsell/             # ── SubAgent 4: Cross-Sell & Up-Sell
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── affinity_analyzer.py   # Co-occurrence patterns
│   │   │   ├── recommender.py         # LLM recommendation engine
│   │   │   ├── inventory_checker.py   # Stock availability
│   │   │   └── ranker.py              # Multi-factor scoring
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_agent.py
│   │
│   ├── menu_optimization/             # ── SubAgent 5: Menu Engineering
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── profitability_analyzer.py  # Stars/Plowhorses/Puzzles/Dogs
│   │   │   ├── price_optimizer.py         # Dynamic pricing suggestions
│   │   │   ├── menu_layout_advisor.py     # Placement recommendations
│   │   │   └── seasonal_planner.py        # Seasonal menu rotation
│   │   ├── schemas/
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       └── test_agent.py
│   │
│   ├── supply_chain_forecast/         # ── SubAgent 6: Supply & Demand
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── demand_forecaster.py   # Predict ingredient needs
│   │   │   ├── waste_tracker.py       # Food waste analytics
│   │   │   ├── supplier_matcher.py    # RAG over supplier database
│   │   │   └── cost_analyzer.py       # Ingredient cost trends
│   │   ├── schemas/
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       └── test_agent.py
│   │
│   ├── customer_feedback/             # ── SubAgent 7: Feedback & Sentiment
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── sentiment_analyzer.py  # Review sentiment classification
│   │   │   ├── complaint_router.py    # Auto-route complaints
│   │   │   ├── feedback_summarizer.py # Aggregate trends from reviews
│   │   │   └── nps_calculator.py      # Net Promoter Score engine
│   │   ├── schemas/
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       └── test_agent.py
│   │
│   ├── nutrition_compliance/          # ── SubAgent 8: Nutrition & Compliance
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── nutrition_calculator.py # Macro/micro nutrient analysis
│   │   │   ├── allergen_checker.py     # Cross-contamination detection
│   │   │   ├── label_generator.py      # Regulatory label creation
│   │   │   └── compliance_validator.py # FDA/FSSAI/EU regulation check
│   │   ├── schemas/
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       └── test_agent.py
│   │
│   ├── kitchen_operations/            # ── SubAgent 9: Kitchen Ops
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── agent.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── prep_scheduler.py      # Prep list generation
│   │   │   ├── recipe_scaler.py       # Scale recipes by servings
│   │   │   ├── equipment_planner.py   # Station/equipment allocation
│   │   │   └── sop_generator.py       # Standard operating procedures
│   │   ├── schemas/
│   │   │   └── models.py
│   │   ├── data/
│   │   │   └── seed.json
│   │   └── tests/
│   │       └── test_agent.py
│   │
│   └── marketing_campaigns/           # ── SubAgent 10: Marketing & Promos
│       ├── __init__.py
│       ├── README.md
│       ├── agent.py
│       ├── state.py
│       ├── prompts.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── campaign_generator.py  # AI copy for campaigns
│       │   ├── audience_segmenter.py  # Customer segmentation
│       │   ├── channel_optimizer.py   # Best channel recommendation
│       │   └── roi_predictor.py       # Campaign ROI estimation
│       ├── schemas/
│       │   └── models.py
│       ├── data/
│       │   └── seed.json
│       └── tests/
│           └── test_agent.py
│
├── vectorstore/
│   ├── __init__.py
│   ├── chroma_client.py               # ChromaDB initialization
│   ├── collections.py                 # Collection management
│   └── ingestion.py                   # Document chunking + embedding
│
├── api/
│   ├── __init__.py
│   ├── app.py                         # FastAPI app factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py                    # POST /api/chat — main endpoint
│   │   ├── health.py                  # GET /health
│   │   └── admin.py                   # Agent registry, reload
│   └── middleware/
│       ├── __init__.py
│       ├── rate_limiter.py
│       └── request_logger.py
│
├── frontend/
│   ├── index.html                     # Chat UI
│   ├── style.css
│   └── app.js
│
├── data/
│   ├── chroma_db/                     # ChromaDB persistent storage
│   └── seed/                          # Shared seed data
│       └── products.json
│
├── scripts/
│   ├── seed_vectorstore.py            # Ingest seed data to ChromaDB
│   └── run_tests.sh                   # Test runner
│
├── docs/
│   ├── ARCHITECTURE.md                # System design deep-dive
│   ├── FLOW_DIAGRAM.md                # Mermaid diagrams
│   └── SUBAGENT_TEMPLATE.md           # Blank template for new agents
│
└── tests/
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures
    ├── test_orchestrator.py
    ├── test_guardrails.py
    └── test_search_service.py
```

---

## Part 2: Complete Flow — How the System Works

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Reformation   │ ← Rewrite for clarity, expand abbreviations
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Input Guardrail     │ ← Regex: PII, profanity, SQL injection
│  (Regex + Agent)     │   Agent: topic relevance (F&B only)
└─────────┬───────────┘
          │ PASS
          ▼
┌─────────────────────┐
│  Orchestrator        │ ← Intent classification via LLM
│  (Supervisor Node)   │   Determines which SubAgent(s) to call
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Execution Strategy  │ ← Parallel if multiple agents needed
│                      │   Sequential if dependencies exist
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────┐
│  Pre-Tool Guardrail                      │ ← Validate tool params
│  (before each subgraph runs)             │   Check authorization
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────────────────────────┐
│  SubGraph Execution                      │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ If RAG tool  │→│ Search Service   │   │
│  │ needed       │ │ (ChromaDB)       │   │
│  └─────────────┘  └────────┬────────┘   │
│                            ▼             │
│                   ┌─────────────────┐    │
│                   │Search Evaluation│    │  ← Is retrieval good enough?
│                   └────────┬────────┘    │    If not → reformulate & retry
│                            ▼             │
│  ┌─────────────────────────────────┐     │
│  │ Tool Response Collection         │     │
│  └─────────────────────────────────┘     │
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────┐
│ Response Aggregation │ ← Merge results from multiple tools/agents
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Answer Evaluation    │ ← Score: relevance, completeness, accuracy
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Output Guardrail     │ ← No hallucinations, no PII in response
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Tone of Voice Check  │ ← Brand voice: professional, friendly, F&B
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Final Response       │ ← Answer + Citations + Time Taken
│ + Citations + Timer  │
└─────────────────────┘
```

---

## Part 3: The Claude Code Prompt

Copy this ENTIRE prompt into Claude Code. Run it from the root of your project directory.

---

```
You are building a production-grade multi-agent system for the Food & Beverage industry.

PROJECT: fnb-agentic-platform
TECH: Python 3.11, LangGraph (StateGraph + subgraphs), LangChain, Pydantic v2, FastAPI, ChromaDB, OpenAI GPT-4o-mini
STYLE: Clean, typed, documented. Every function has docstrings. Every model uses Pydantic.

═══════════════════════════════════════════════════
STEP 1: CREATE THE COMPLETE FOLDER STRUCTURE
═══════════════════════════════════════════════════

Create every folder and __init__.py file for this structure. Leave subagent folders for 10 subagents:
product_ideation, product_catalog, customer_loyalty, cross_sell_upsell, menu_optimization, supply_chain_forecast, customer_feedback, nutrition_compliance, kitchen_operations, marketing_campaigns

Each subagent folder must have: __init__.py, README.md, agent.py, state.py, prompts.py, tools/, schemas/, data/, tests/

═══════════════════════════════════════════════════
STEP 2: CORE FRAMEWORK FILES (config/, core/)
═══════════════════════════════════════════════════

FILE: config/settings.py
- Use pydantic_settings.BaseSettings
- Fields: OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o-mini), CHROMA_PERSIST_DIR (default ./data/chroma_db), APP_NAME, ENVIRONMENT, LOG_LEVEL, API_HOST, API_PORT (8000), DATABASE_URL
- Load from .env via model_config

FILE: config/constants.py
- MAX_RETRIES = 3, SEARCH_TOP_K = 5, MAX_CONTEXT_TOKENS = 4000
- GUARDRAIL_BLOCKED_PATTERNS: list of regex patterns (SQL injection, PII patterns like SSN/credit card, profanity placeholder)
- BRAND_TONE_INSTRUCTIONS: str with F&B professional tone guidance
- SUBAGENT_NAMES: list of all 10 subagent string identifiers

FILE: core/models/base_state.py
- Use TypedDict for AgentState with fields:
  messages: Annotated[list, add_messages]
  query_original: str
  query_reformed: str
  intent: str
  target_agents: list[str]
  guardrail_status: str  # "pass" | "blocked"
  guardrail_reason: str
  tool_responses: list[dict]
  aggregated_response: str
  answer_score: float
  tone_check_passed: bool
  citations: list[dict]
  execution_time_ms: float
  final_response: str

FILE: core/models/tool_schemas.py
- ToolInput(BaseModel): query: str, context: dict = {}, parameters: dict = {}
- ToolOutput(BaseModel): success: bool, data: Any, source: str, confidence: float, execution_time_ms: float
- SearchResult(BaseModel): content: str, metadata: dict, score: float, source: str

FILE: core/models/api_schemas.py
- ChatRequest(BaseModel): message: str, session_id: str = Field(default_factory=lambda: str(uuid4())), customer_id: str | None = None
- ChatResponse(BaseModel): response: str, intent: str, agents_used: list[str], citations: list[dict], execution_time_ms: float, session_id: str

═══════════════════════════════════════════════════
STEP 3: GUARDRAILS (core/guardrails/)
═══════════════════════════════════════════════════

FILE: core/guardrails/regex_guardrail.py
- Function: check_regex_guardrails(text: str) -> tuple[bool, str]
- Check against GUARDRAIL_BLOCKED_PATTERNS from constants
- Return (True, "") if clean, (False, "reason") if blocked
- Check for: SQL injection patterns, PII (SSN, credit card, email extraction attempts), prompt injection attempts (ignore previous, system prompt, etc.)

FILE: core/guardrails/agent_guardrail.py
- Function: async check_agent_guardrail(query: str, llm) -> tuple[bool, str]
- Use LLM to classify if query is related to F&B domain
- System prompt: "You are a guardrail. Determine if this query is related to food, beverage, restaurant, or hospitality industry. Respond ONLY with JSON: {\"allowed\": true/false, \"reason\": \"...\"}"
- Parse response, return tuple

FILE: core/guardrails/pre_tool_guardrail.py
- Function: check_pre_tool(tool_name: str, tool_input: ToolInput) -> tuple[bool, str]
- Validate tool inputs are within expected bounds
- Check that tool_name exists in registry

FILE: core/guardrails/output_guardrail.py
- Function: async check_output_guardrail(response: str, llm) -> tuple[bool, str]
- LLM check: no hallucinated data, no PII leaked, response is professional
- Return pass/fail

═══════════════════════════════════════════════════
STEP 4: QUERY REFORMATION (core/query/)
═══════════════════════════════════════════════════

FILE: core/query/reformulator.py
- Function: async reform_query(query: str, chat_history: list, llm) -> str
- System prompt: "You are a query reformulator for an F&B AI assistant. Rewrite the user query to be clear, specific, and self-contained. Expand abbreviations. Resolve pronouns using chat history. Output ONLY the reformulated query."
- If query is already clear, return as-is

═══════════════════════════════════════════════════
STEP 5: SEARCH SERVICE (core/search/)
═══════════════════════════════════════════════════

FILE: core/search/embeddings.py
- Class: EmbeddingService
- Uses OpenAI text-embedding-3-small
- Method: embed_text(text: str) -> list[float]
- Method: embed_batch(texts: list[str]) -> list[list[float]]

FILE: core/search/search_service.py
- Class: SearchService
- __init__: Initialize ChromaDB client (PersistentClient), get or create collection
- Method: async search(query: str, collection_name: str, top_k: int = 5, filters: dict = None) -> list[SearchResult]
- Method: add_documents(documents: list[str], metadatas: list[dict], collection_name: str)
- Use ChromaDB's built-in embedding or OpenAI embeddings

FILE: core/search/search_evaluator.py
- Function: evaluate_search_results(query: str, results: list[SearchResult]) -> tuple[bool, float]
- Check: at least 1 result above 0.7 similarity threshold
- Check: results are relevant (not empty, have content)
- Return (is_good_enough: bool, avg_score: float)

═══════════════════════════════════════════════════
STEP 6: RESPONSE PIPELINE (core/response/)
═══════════════════════════════════════════════════

FILE: core/response/aggregator.py
- Function: aggregate_responses(tool_responses: list[dict]) -> str
- Merge responses from multiple tools into coherent context
- Deduplicate information
- Format as structured input for answer generation

FILE: core/response/answer_evaluator.py
- Function: async evaluate_answer(query: str, answer: str, llm) -> float
- LLM scores answer 0.0-1.0 on: relevance, completeness, accuracy
- Returns float score

FILE: core/response/tone_checker.py
- Function: async check_tone(response: str, llm) -> tuple[bool, str]
- Verify response matches brand voice: professional, helpful, F&B-knowledgeable
- If fails, return corrected version

FILE: core/response/citation_builder.py
- Function: build_citations(tool_responses: list[dict]) -> list[dict]
- Extract source info from each tool response
- Format: [{"source": "...", "tool": "...", "confidence": 0.95}]
- Function: format_final_response(answer: str, citations: list[dict], time_ms: float) -> str
- Append citations and execution time to response

═══════════════════════════════════════════════════
STEP 7: ORCHESTRATOR (core/orchestrator/)
═══════════════════════════════════════════════════

FILE: core/orchestrator/state.py
- Re-export AgentState from core.models.base_state
- Add any orchestrator-specific state helpers

FILE: core/orchestrator/router.py
- Function: async route_intent(state: AgentState, llm) -> AgentState
- Use LLM to classify intent from reformed query
- System prompt lists all 10 subagent names and their descriptions
- Output JSON: {"agents": ["product_catalog", "cross_sell_upsell"], "intent": "product recommendation with upsell", "reasoning": "..."}
- Update state with target_agents and intent

FILE: core/orchestrator/execution_strategy.py
- Function: determine_strategy(target_agents: list[str]) -> str
- If single agent: return "sequential"
- If multiple independent: return "parallel"
- If dependencies: return "sequential"
- Function: async execute_agents(state: AgentState, subagent_registry: dict, strategy: str) -> AgentState
- Run subgraphs based on strategy, collect tool_responses

FILE: core/orchestrator/main_graph.py
- Build the main LangGraph StateGraph
- Nodes (in this order):
  1. "query_reform" → calls reformulator
  2. "input_guardrail" → calls regex + agent guardrail
  3. "router" → calls route_intent
  4. "execute" → calls execution_strategy (which invokes subgraphs)
  5. "aggregate" → calls aggregator
  6. "evaluate_answer" → calls answer_evaluator
  7. "output_guardrail" → calls output guardrail
  8. "tone_check" → calls tone_checker
  9. "finalize" → calls citation_builder, sets final_response
- Conditional edges:
  - After "input_guardrail": if blocked → go to "finalize" with blocked message
  - After "output_guardrail": if blocked → rewrite and re-evaluate (max 2 retries)
- Compile with checkpointer=None (stateless per request)
- Export: create_main_graph() -> CompiledGraph

═══════════════════════════════════════════════════
STEP 8: SUBAGENT TEMPLATE (every subagent follows this)
═══════════════════════════════════════════════════

Each subagent's agent.py must follow this exact pattern:

```python
"""
SubAgent: [Name]
Purpose: [One-liner]
"""
from langgraph.graph import StateGraph, START, END
from core.models.base_state import AgentState
from core.guardrails.pre_tool_guardrail import check_pre_tool
from core.search.search_service import SearchService
from core.search.search_evaluator import evaluate_search_results

class [Name]Agent:
    def __init__(self, llm, search_service: SearchService):
        self.llm = llm
        self.search = search_service
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("pre_guardrail", self._pre_guardrail)
        workflow.add_node("search", self._search_knowledge)
        workflow.add_node("evaluate_search", self._evaluate_search)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("collect_response", self._collect_response)

        workflow.add_edge(START, "pre_guardrail")
        workflow.add_conditional_edges("pre_guardrail", self._should_proceed)
        workflow.add_edge("search", "evaluate_search")
        workflow.add_conditional_edges("evaluate_search", self._search_good_enough)
        workflow.add_edge("execute_tools", "collect_response")
        workflow.add_edge("collect_response", END)

        return workflow.compile()

    async def _pre_guardrail(self, state): ...
    async def _search_knowledge(self, state): ...  # ChromaDB RAG
    async def _evaluate_search(self, state): ...   # Quality check
    async def _execute_tools(self, state): ...     # Tool logic
    async def _collect_response(self, state): ...  # Package results

    async def run(self, state: AgentState) -> AgentState:
        return await self.graph.ainvoke(state)
```

FILE: subagents/registry.py
- Class: SubAgentRegistry
- Method: register(name: str, agent_class) — stores agent class
- Method: get(name: str) -> agent instance
- Method: list_agents() -> list of registered names
- Auto-discover: on init, scan subagents/ for agent.py files and register them
- Each subagent's __init__.py should export: AGENT_NAME: str, DESCRIPTION: str, AgentClass

═══════════════════════════════════════════════════
STEP 9: VECTORSTORE (vectorstore/)
═══════════════════════════════════════════════════

FILE: vectorstore/chroma_client.py
- Function: get_chroma_client() -> chromadb.PersistentClient
- Use settings.CHROMA_PERSIST_DIR

FILE: vectorstore/collections.py
- One collection per subagent: "product_ideation", "product_catalog", etc.
- Function: get_or_create_collection(name: str) -> Collection

FILE: vectorstore/ingestion.py
- Function: ingest_documents(file_path: str, collection_name: str, chunk_size: int = 500, chunk_overlap: int = 50)
- Use RecursiveCharacterTextSplitter from langchain
- Add metadata: source, chunk_index, timestamp

═══════════════════════════════════════════════════
STEP 10: API LAYER (api/)
═══════════════════════════════════════════════════

FILE: api/app.py
- FastAPI app factory: create_app() -> FastAPI
- Include CORS, mount static files (frontend/), include routers
- On startup: initialize LLM, SearchService, SubAgentRegistry, compile main_graph

FILE: api/routes/chat.py
- POST /api/chat — accepts ChatRequest, runs main_graph, returns ChatResponse
- Track total execution time
- Handle errors gracefully

FILE: api/routes/health.py
- GET /health — return {"status": "healthy", "agents_loaded": [...], "version": "1.0.0"}

FILE: api/routes/admin.py
- GET /api/agents — list all registered subagents
- POST /api/agents/reload — re-scan and reload registry

═══════════════════════════════════════════════════
STEP 11: ENTRY POINT + CONFIG FILES
═══════════════════════════════════════════════════

FILE: main.py
- import uvicorn, create_app
- if __name__ == "__main__": uvicorn.run(...)

FILE: requirements.txt
langgraph>=0.2.50
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.6.0
chromadb>=0.5.0
python-dotenv>=1.0.0
httpx>=0.27.0
structlog>=24.0.0
tiktoken>=0.7.0

FILE: .env.example
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
APP_NAME=FnB Agentic Platform
ENVIRONMENT=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
CHROMA_PERSIST_DIR=./data/chroma_db

FILE: .gitignore
(standard Python + .env + data/chroma_db/ + __pycache__/ + venv/ + .vscode/ + *.pyc)

FILE: Dockerfile
FROM python:3.11-slim, WORKDIR /app, COPY requirements.txt, pip install, COPY ., EXPOSE 8000, CMD uvicorn

FILE: docker-compose.yml
Single service: fnb-platform, build ., ports 8000:8000, env_file .env, volumes for data persistence

═══════════════════════════════════════════════════
STEP 12: STUB OUT ALL 10 SUBAGENTS
═══════════════════════════════════════════════════

For each of the 10 subagents, create:
1. agent.py — with the subgraph pattern from Step 8 (working stub with TODO comments)
2. state.py — SubAgent-specific state extending AgentState
3. prompts.py — System prompt tailored to that domain
4. tools/ — 3-4 tool files with function signatures, docstrings, and placeholder implementations returning mock data
5. schemas/models.py — Pydantic models relevant to that domain
6. data/seed.json — 5-10 sample records
7. tests/test_agent.py — Basic test that agent initializes and runs with mock input
8. README.md — What this agent does, tools available, how to extend, sample queries
9. __init__.py — exports AGENT_NAME, DESCRIPTION, AgentClass

SUBAGENT DETAILS:

1. product_ideation: Trend research (web search), concept generation, competitor analysis, idea validation
2. product_catalog: Product CRUD, RAG search, AI pairings, bundle creation, allergen/nutrition lookup
3. customer_loyalty: Points earn/burn, tier management, digital wallet, offers/missions/coupons
4. cross_sell_upsell: Affinity analysis, LLM recommendations, inventory check, multi-factor ranking
5. menu_optimization: Menu engineering matrix (Stars/Dogs/etc), price optimization, layout advice, seasonal rotation
6. supply_chain_forecast: Demand prediction, waste tracking, supplier matching (RAG), ingredient cost analysis
7. customer_feedback: Sentiment analysis, complaint routing, feedback summarization, NPS calculation
8. nutrition_compliance: Nutrition calculation, allergen detection, label generation, regulatory compliance (FDA/FSSAI/EU)
9. kitchen_operations: Prep scheduling, recipe scaling, equipment planning, SOP generation
10. marketing_campaigns: Campaign copy generation, audience segmentation, channel optimization, ROI prediction

═══════════════════════════════════════════════════
STEP 13: FRONTEND CHAT UI
═══════════════════════════════════════════════════

FILE: frontend/index.html + style.css + app.js
- Clean chat interface, dark theme
- Shows: user messages, assistant responses with citations
- Shows execution time badge on each response
- Shows which agent(s) handled the query
- Input box with send button
- Auto-scroll to latest message

═══════════════════════════════════════════════════
IMPORTANT PATTERNS TO FOLLOW:
═══════════════════════════════════════════════════

1. Every async function that calls LLM must have try/except with structured logging
2. Every tool returns ToolOutput schema
3. Subgraphs are compiled StateGraphs, not just functions
4. ChromaDB is used for all RAG operations — no FAISS, no Pinecone
5. All prompts are in prompts.py, never hardcoded in logic
6. Use structlog for all logging with context (agent_name, tool_name, execution_time)
7. Every file has module-level docstring explaining its purpose
8. Type hints on every function signature
9. No global state — everything injected via constructor or function params
```

---

## Part 4: SubAgent Summary Table (For Student Assignment)

| # | SubAgent | Domain | Key Tools | RAG Collection |
|---|----------|--------|-----------|----------------|
| 1 | Product Ideation | R&D / Innovation | Trend Researcher, Concept Generator, Competitor Analyzer, Idea Validator | `product_ideation` |
| 2 | Product Catalog | Menu Management | Catalog Search (RAG), CRUD, Pairing Engine, Bundle Creator | `product_catalog` |
| 3 | Customer Loyalty | CRM / Loyalty | Points Engine, Tier Manager, Wallet Manager, Offer Engine | `customer_loyalty` |
| 4 | Cross-Sell & Up-Sell | Revenue Growth | Affinity Analyzer, Recommender, Inventory Checker, Ranker | `cross_sell_upsell` |
| 5 | Menu Optimization | Menu Engineering | Profitability Analyzer, Price Optimizer, Layout Advisor, Seasonal Planner | `menu_optimization` |
| 6 | Supply Chain Forecast | Operations | Demand Forecaster, Waste Tracker, Supplier Matcher (RAG), Cost Analyzer | `supply_chain_forecast` |
| 7 | Customer Feedback | Voice of Customer | Sentiment Analyzer, Complaint Router, Feedback Summarizer, NPS Calculator | `customer_feedback` |
| 8 | Nutrition & Compliance | Regulatory | Nutrition Calculator, Allergen Checker, Label Generator, Compliance Validator | `nutrition_compliance` |
| 9 | Kitchen Operations | Back-of-House | Prep Scheduler, Recipe Scaler, Equipment Planner, SOP Generator | `kitchen_operations` |
| 10 | Marketing Campaigns | Marketing | Campaign Generator, Audience Segmenter, Channel Optimizer, ROI Predictor | `marketing_campaigns` |

---

## Part 5: Student Git Workflow Guide (CONTRIBUTING.md)

### Prerequisites

1. Git installed and configured (`git config --global user.name "Your Name"`)
2. GitHub account
3. Python 3.11+ installed
4. Access to the main repository (instructor will share the URL)

### Step-by-Step Workflow

#### 1. Fork the Repository

Go to the main repo on GitHub → Click **Fork** → This creates YOUR copy.

#### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/fnb-agentic-platform.git
cd fnb-agentic-platform
```

#### 3. Add Upstream Remote

```bash
git remote add upstream https://github.com/INSTRUCTOR_ORG/fnb-agentic-platform.git
git remote -v   # Verify: origin = yours, upstream = instructor's
```

#### 4. Create Your Feature Branch

**Branch naming convention:** `feature/<your-name>/<subagent-name>`

```bash
git checkout -b feature/john/menu-optimization
```

#### 5. Set Up Your Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

#### 6. Work ONLY in Your SubAgent Folder

You are only allowed to modify files inside:
```
subagents/<your-assigned-subagent>/
```

**Do NOT modify** anything in `core/`, `config/`, `api/`, or other subagent folders.

#### 7. What You Must Implement

| File | What to do |
|------|-----------|
| `agent.py` | Replace TODO stubs with working LangGraph subgraph logic |
| `state.py` | Add any domain-specific state fields you need |
| `prompts.py` | Write production-quality system prompts |
| `tools/*.py` | Implement each tool with real logic (API calls, LLM calls, DB queries) |
| `schemas/models.py` | Define all Pydantic models your tools need |
| `data/seed.json` | Add realistic F&B sample data (minimum 20 records) |
| `tests/test_agent.py` | Write at least 5 test cases (happy path + edge cases) |
| `README.md` | Document your agent: purpose, tools, sample queries, architecture decisions |

#### 8. Commit Conventions

```bash
# Format: <type>(<scope>): <description>
git add subagents/menu_optimization/
git commit -m "feat(menu-optimization): implement profitability analyzer tool"
git commit -m "test(menu-optimization): add edge case tests for price optimizer"
git commit -m "docs(menu-optimization): update README with architecture diagram"
```

**Types:** `feat`, `fix`, `test`, `docs`, `refactor`, `style`

#### 9. Push and Create Merge Request

```bash
# Sync with upstream first
git fetch upstream
git rebase upstream/main

# Push your branch
git push origin feature/john/menu-optimization
```

Go to GitHub → Your fork → Click **"Compare & pull request"**

**MR Title:** `[SubAgent] Menu Optimization — Complete Implementation`

**MR Description Template:**
```markdown
## SubAgent: Menu Optimization

### What's Implemented
- [ ] agent.py — Full subgraph with 4 nodes
- [ ] tools/profitability_analyzer.py — Menu engineering matrix
- [ ] tools/price_optimizer.py — Elasticity-based pricing
- [ ] tools/menu_layout_advisor.py — Position recommendations
- [ ] tools/seasonal_planner.py — Rotation calendar
- [ ] schemas/models.py — All Pydantic models
- [ ] data/seed.json — 25 sample menu items
- [ ] tests/test_agent.py — 8 test cases (all passing)
- [ ] README.md — Complete documentation

### Sample Queries That Work
1. "Which menu items are underperforming?"
2. "Suggest optimal pricing for our burger range"
3. "What should our summer menu look like?"

### How to Test
```bash
pytest subagents/menu_optimization/tests/ -v
```

### Screenshots
(Chat UI screenshots showing agent in action)
```

#### 10. Respond to Code Review

The instructor will review your MR. Address feedback, push fixes:
```bash
git add .
git commit -m "fix(menu-optimization): address review — improve error handling"
git push origin feature/john/menu-optimization
```

### Rules

1. **One student per subagent** — no overlapping work
2. **Never modify core/** — if you need a shared utility, propose it in a separate MR
3. **All tests must pass** before submitting MR
4. **Minimum 5 test cases** per subagent
5. **Seed data must be realistic** — real F&B product names, prices, categories
6. **No API keys in code** — everything via .env
7. **Document everything** — future students will read your code

---

## Part 6: Testing Checklist for Students

Before submitting your MR, verify:

```
□ python -m pytest subagents/<your-agent>/tests/ -v  → All pass
□ Your agent initializes without errors
□ At least 3 sample queries produce correct responses
□ RAG search returns relevant results from your seed data
□ Guardrails block off-topic queries
□ All tools return ToolOutput schema
□ No hardcoded API keys or secrets
□ README.md has: purpose, architecture, tools, sample queries
□ Code has type hints and docstrings
□ Commit messages follow convention
```
