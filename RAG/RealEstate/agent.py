from __future__ import annotations

import json
import operator
import os
import config
import applogging
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE)
logger = applogging.get_logger("real_estate_app")

class RealEstateState(BaseModel):
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

    # health_view: str = ""
    # gym_view: str = ""
    # fitness_view: str = ""

    needs_detailed_answer: bool = False
    answer_reason: str = ""
    final_answer: str = ""

    messages: Annotated[list[str], operator.add] = []

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

def search_index(state: RealEstateState) -> dict:
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

def understand_question(state: RealEstateState) -> dict:
    """
    First node: interpret the user's question before retrieval.
    """
    response = llm.invoke(
        f"You are a helpful real estate assistant who has knowledge of current real estate trends, real estate investments etc.\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"In 2-3 short sentences, analyze and explain what the user is expecting as an answer of the query/question submitted.\n"
        f"Mention whether the question is mainly about real estate investment prospective or need some guidance about current trends "
        f"in real estate markets or user is mainly looking for some guide line as a buyer or seller any properties."
    )

    return {
        "question_analysis": response.content,
        "messages": [f"[understand_question] {response.content}"],
    }

def build_real_estate_agent():
    """
    Build and compile the LangGraph application.

    Graph structure:
        START -> understand_question -> search_index
              -> 
              -> 
              -> 
              -> pick_response_mode
              -> quick_answer OR detailed_answer
              -> END
    """
    graph = StateGraph(RealEstateState)

    graph.add_node("understand_question", understand_question)
    graph.add_node("search_index", search_index)
    # graph.add_node("health_specialist", health_specialist)
    # graph.add_node("gym_specialist", gym_specialist)
    # graph.add_node("fitness_specialist", fitness_specialist)
    # graph.add_node("pick_response_mode", pick_response_mode)
    # graph.add_node("quick_answer", quick_answer)
    # graph.add_node("detailed_answer", detailed_answer)

    graph.add_edge(START, "understand_question")
    graph.add_edge("understand_question", "search_index")

    # graph.add_edge("search_index", "health_specialist")
    # graph.add_edge("search_index", "gym_specialist")
    # graph.add_edge("search_index", "fitness_specialist")

    # graph.add_edge("health_specialist", "pick_response_mode")
    # graph.add_edge("gym_specialist", "pick_response_mode")
    # graph.add_edge("fitness_specialist", "pick_response_mode")

    # graph.add_conditional_edges(
    #     "pick_response_mode",
    #     route_after_decision,
    #     {
    #         "quick": "quick_answer",
    #         "detailed": "detailed_answer",
    #     },
    # )

    # graph.add_edge("quick_answer", END)
    # graph.add_edge("detailed_answer", END)

    graph.add_edge("search_index", END)

    return graph.compile()

def query_rag(question: str) -> str:
    """Run one user question through the health and fitness LangGraph agent."""
    app = build_real_estate_agent()
    result = app.invoke(
        {
            "user_question": question,
            "messages": [],
        }
    )
    #return result["final_answer"]
    return result["question_analysis"]

if __name__ == "__main__":
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
    answer = query_rag("Tell me how is the real estate market in 2026.")
    print(answer)