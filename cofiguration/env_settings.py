import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Dynamically find your workspace root directory
# __file__ is 'your_workspace_root/configuration/settings.py'
# .parent is 'configuration/', .parent.parent is 'your_workspace_root/'
SETTINGS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = SETTINGS_FILE.parent.parent

# 2. Load the .env file explicitly from the root folder
load_dotenv(dotenv_path=WORKSPACE_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_RESOURCE_ENDPOINT = os.getenv("AZURE_RESOURCE_ENDPOINT")
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
