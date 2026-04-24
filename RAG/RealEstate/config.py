import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
DEBUG = bool(os.getenv("DEBUG", False))
TOP_K = int(os.getenv("TOP_K", 4))
TEMPERATURE = int(os.getenv("TEMPERATURE", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))