import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR")
LLM_MODEL = os.getenv("LLM_MODEL")
DEBUG = bool(os.getenv("DEBUG", False))
TOP_K = int(os.getenv("TOP_K", 4))
# Temperature should be a float between 0.0 and 1.0. Use 0.0 for deterministic outputs.
try:
	TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
except (TypeError, ValueError):
	TEMPERATURE = 0.0
MCP_SERVERS_DIR = os.getenv("MCP_SERVERS_DIR", "mcp_servers")	
