"""Shared LLM instance. Single point of control for all nodes."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config import LLM_MODEL, LLM_TEMPERATURE

load_dotenv()

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=os.getenv("OPENAI_API_KEY"),
)
