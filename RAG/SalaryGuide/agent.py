from __future__ import annotations

import json
import operator
import os
import config
import applogging
from typing import Annotated

from pydantic import BaseModel, ConfigDict

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE)
logger = applogging.get_logger("salary_guide_app")



class SalaryGuideState(BaseModel):
    """
    Shared state that flows through the LangGraph application.

    Students can read this class top-to-bottom to understand what data each
    node produces and consumes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_question: str = ""
    question_analysis: str = ""

    retrieved_documents: list[Document] = []
    retrieved_context: str = ""
    retrieved_sources: str = ""

    salary_analysis: str = ""

    messages: Annotated[list[str], operator.add] = []

def hr_specialist(state: SalaryGuideState) -> dict:
    '''Parallel node: ask the HR specialist to analyze the retrieved context and answer the user's question.'''
    response = llm.invoke(
        f"You are a human resource manager who specializes in employee compensation and benefits.\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"Using only the retrieved context:\n{state.retrieved_context}\n\n"
        f"Extract Bill Rate, Target Range, and Exception Range values and respond to the user's question in clear language."
    )

    return {
        "salary_analysis": response.content,
        "messages": [f"[hr_specialist] DONE"],
    }

def format_context(documents: list[Document]) -> str:
    """Combine retrieved chunks into one prompt-ready context string."""
    if not documents:
        return "No relevant context was retrieved from the index."

    return "\n\n---\n\n".join(document.page_content for document in documents)

def format_sources(documents: list[Document]) -> str:
    """Format citation metadata into a readable source list."""
    if not documents:
        return "No sources retrieved."

    formatted_sources = []
    for index, document in enumerate(documents, start=1):
        source_file = document.metadata.get("source", "Unknown source")
        page_number = document.metadata.get("page", "?")
        page_label = page_number + 1 if isinstance(page_number, int) else page_number
        formatted_sources.append(f"[{index}] {source_file} (Page {page_label})")

    return "\n".join(formatted_sources)

def build_embedding_model() -> HuggingFaceEmbeddings:
    """Create the local embedding model used to search Chroma."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def load_vector_store() -> Chroma:
    """Open the local Chroma vector database from disk."""
    if not os.path.exists(config.CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"Vector database '{config.CHROMA_DB_DIR}/' was not found. Run ingestion first."
        )

    return Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=build_embedding_model(),
    )

def search_index(state: SalaryGuideState) -> dict:
    """
    Retrieval node: search the Chroma index for relevant chunks.

    This is the key node you wanted to show your students explicitly.
    """
    vector_store = load_vector_store()
    retrieved_documents = vector_store.similarity_search(state.user_question, k=config.TOP_K)

    retrieved_context = format_context(retrieved_documents)
    retrieved_sources = format_sources(retrieved_documents)

    logger.info(f"[search_index] Found {len(retrieved_documents)} chunk(s)")
    return {
        "retrieved_documents": retrieved_documents,
        "retrieved_context": retrieved_context,
        "retrieved_sources": retrieved_sources,
        "messages": [f"[search_index] Retrieved {len(retrieved_documents)} chunk(s)"],
    }

def understand_question(state: SalaryGuideState) -> dict:
    """
    First node: interpret the user's question before retrieval.
    """
    response = llm.invoke(
        f"You are company's Human Resource assistant who has access of knowledge based documents.\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"From user's question, analyze labor category (LCAT), employment type (1099 or w2), and company name.\n"
        f"Search the salary guide PDF for matching entries."
    )

    return {
        "question_analysis": response.content,
        "messages": [f"[understand_question] {response.content}"],
    }

def build_salary_guide_agent():

    graph = StateGraph(SalaryGuideState)

    graph.add_node("understand_question", understand_question)
    graph.add_node("search_index", search_index)
    graph.add_node("hr_specialist", hr_specialist)

    graph.add_edge(START, "understand_question")
    graph.add_edge("understand_question", "search_index")
    graph.add_edge("search_index", "hr_specialist")
    graph.add_edge("hr_specialist", END)

    return graph.compile()

def query_rag(question: str) -> dict:
    """Run one user question through the real estate LangGraph agent.

    Returns a dict with keys:
    - salary_analysis: str
    - retrieved_sources: str (optional)
    """
    app = build_salary_guide_agent()
    result = app.invoke({"user_question": question, "messages": []})
    return {
        "salary_analysis": result.get("salary_analysis", ""),
        "retrieved_sources": result.get("retrieved_sources", ""),
    }

if __name__ == "__main__":
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
    answer = query_rag("Tell me 1099 bill rate for LCAT DBE3 Leidos.")
    print(answer)
