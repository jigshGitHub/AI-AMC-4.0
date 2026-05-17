"""Single source of truth for absolute paths and config values."""
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent

SEED_DATA_DIR: Path = PROJECT_ROOT / "seed_data"
CUISINES_MD_PATH: Path = SEED_DATA_DIR / "cuisines.md"
CHROMA_DB_PATH: Path = PROJECT_ROOT / "chroma_db"

DUCKDUCKGO_MCP_COMMAND: str = "uvx"
DUCKDUCKGO_MCP_ARGS: list[str] = ["duckduckgo-mcp-server"]

CHROMA_COLLECTION_NAME: str = "cuisines"
RAG_TOP_K: int = 4

SEARCH_EVAL_THRESHOLD: float = 0.5
ANSWER_EVAL_THRESHOLD: float = 0.6
MAX_RETRIES_PER_SUBGRAPH: int = 1

LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0
