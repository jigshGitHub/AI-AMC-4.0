import os
import sys
import math
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Callable, Set

load_dotenv()
def getOpenAIClient():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def getEmbeddingModel():
    return os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-small")

def getChromaDBDir():
    return os.getenv("CHROMA_DB_DIR", "./chroma_db")

envsettings : Set[Callable[..., Any]] = {
     getOpenAIClient,
     getEmbeddingModel,
     getChromaDBDir,
}
