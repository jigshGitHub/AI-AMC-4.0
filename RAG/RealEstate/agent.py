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
logger = applogging.get_logger("real_estate_app")


def _llm_text(response) -> str:
    """Normalize different LLM wrapper responses to a plain text string.

    Different SDKs expose the model output in different attributes. This
    helper tries common attribute names and falls back to str(response).
    """
    if response is None:
        return ""
    # Common attribute names used by different wrappers
    for attr in ("content", "text", "response", "data"):
        val = getattr(response, attr, None)
        if isinstance(val, str) and val:
            return val
        # sometimes content is a list or dict
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict) and "text" in first:
                return first.get("text", "")

    # fallback: try string conversion
    try:
        return str(response)
    except Exception:
        return ""

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

    market_analysis: str = ""
    property_insights: str = ""
    investment_strategy: str = ""

    needs_detailed_answer: bool = False
    answer_reason: str = ""
    final_answer: str = ""

    messages: Annotated[list[str], operator.add] = []

def detailed_answer(state: RealEstateState) -> dict:
    """Create a more structured coaching-style answer for deeper questions."""
    response = llm.invoke(
        f"You are a helpful real estate agent.\n"
        f"Answer the user's question using only the information below.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"Question analysis:\n{state.question_analysis}\n\n"
        f"MARKET TREND:\n{state.market_analysis}\n\n"
        f"PROPERTY INSIGHTS:\n{state.property_insights}\n\n"
        f"INVESTMENT STRATEGY:\n{state.investment_strategy}\n\n"
        f"SOURCES:\n{state.retrieved_sources}\n\n"
        f"Write a structured, user-friendly answer with these sections:\n"
        f"1. Main Answer\n"
        f"2. Recommendations\n"
        f"3. Limits or Missing Information\n"
        f"4. Sources\n\n"
        f"If the context is insufficient, say that clearly instead of inventing details."
    )

    return {
        "final_answer": response.content,
        "messages": ["[detailed_answer] Generated detailed answer"],
    }

def quick_answer(state: RealEstateState) -> dict:
    """Create a short answer for straightforward questions."""
    response = llm.invoke(
        f"You are a helpful real estate agent.\n"
        f"Answer the user's question using only the information below.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"MARKET TREND:\n{state.market_analysis}\n\n"
        f"PROPERTY INSIGHTS:\n{state.property_insights}\n\n"
        f"INVESTMENT STRATEGY:\n{state.investment_strategy}\n\n"
        f"SOURCES:\n{state.retrieved_sources}\n\n"
        f"Write a concise, beginner-friendly answer in a short paragraph or a few "
        f"bullets. If the context is insufficient, say so clearly. End with:\n"
        f"Sources:\n"
    )

    return {
        "final_answer": response.content,
        "messages": ["[quick_answer] Generated quick answer"],
    }

def route_after_decision(state: RealEstateState) -> str:
    """Conditional router after the planner node."""
    if state.needs_detailed_answer:
        return "detailed"
    return "quick"

def pick_response_mode(state: RealEstateState) -> dict:
    """
    Fan-in decision node.

    It decides whether the final answer should be:
    - quick: concise explanation
    - detailed: more structured coaching-style response
    """
    response = llm.invoke(
        f"You are a response planner for a real estate RAG assistant.\n\n"
        f"User question:\n{state.user_question}\n\n"
        f"Question analysis:\n{state.question_analysis}\n\n"
        f"MARKET TREND:\n{state.market_analysis}\n\n"
        f"PROPERTY INSIGHTS:\n{state.property_insights}\n\n"
        f"INVESTMENT STRATEGY:\n{state.investment_strategy}\n\n"
        f"Choose whether the user needs a QUICK answer or a DETAILED answer.\n"
        f"Use DETAILED when the user asks for a plan, routine, multi-step guidance in details, "
        f"comparison, or explanation. Use QUICK for straightforward questions.\n\n"
        f"Reply strictly as JSON and nothing else:\n"
        f'{{"needs_detailed_answer": true, "reason": "one sentence"}}'
    )

    # Robust JSON parsing: LLMs may include surrounding text. Extract the
    # first JSON object found in the response and parse it. If parsing fails
    # once, prompt the model to reply with strict JSON and try again.
    raw = _llm_text(response)

    def _extract_json(s: str):
        s = (s or "").strip()
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    parsed = _extract_json(raw)
    needs_detailed_answer = False
    answer_reason = "Could not parse planner output, defaulting to a quick answer."

    if parsed is None:
        # Ask the model to re-output strict JSON only
        retry_prompt = (
            "Please reply strictly with JSON only, with the keys: needs_detailed_answer (true/false) and reason (one sentence).\n"
            "Example: {\"needs_detailed_answer\": true, \"reason\": \"user asked for step-by-step plan\"}"
        )
        retry_resp = llm.invoke(retry_prompt)
        parsed = _extract_json(_llm_text(retry_resp))

    if parsed is not None:
        try:
            needs_detailed_answer = bool(parsed.get("needs_detailed_answer", False))
            answer_reason = str(parsed.get("reason", ""))
        except Exception:
            needs_detailed_answer = False
            answer_reason = "Planner returned unexpected JSON fields."

    return {
        "needs_detailed_answer": needs_detailed_answer,
        "answer_reason": answer_reason,
        "messages": [f"[pick_response_mode] detailed={needs_detailed_answer}"],
    }

def investment_strategy_specialist(state: RealEstateState) -> dict:
    '''Parallel node: ask the investment strategy specialist to analyze the retrieved context and answer the user's question.'''
    response = llm.invoke(
        f"You are a real estate investment strategy specialist who can analyze residential, commercial, and industrial properties. \n\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"Using only the retrieved context:\n{state.retrieved_context}\n\n"
        f"Provide a summary of investment strategy insights and respond to the user's question in clear language."
    )

    return {
        "investment_strategy": response.content,
        "messages": [f"[investment_strategy_specialist] DONE"],
    }

def property_insights_specialist(state: RealEstateState) -> dict:
    '''Parallel node: ask the property insights specialist to analyze the retrieved context and answer the user's question.'''
    response = llm.invoke(
        f"You are a real estate property specialist who can analyze residential, commercial, and industrial properties. \n\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"Using only the retrieved context:\n{state.retrieved_context}\n\n"
        f"Provide a summary of property insights and respond to the user's question in clear language."
    )

    return {
        "property_insights": response.content,
        "messages": [f"[property_insights_specialist] DONE"],
    }

def market_specialist(state: RealEstateState) -> dict:
    '''Parallel node: ask the market specialist to analyze the retrieved context and answer the user's question.'''
    response = llm.invoke(
        f"You are a real estate market analyst with the speciality of analyzing real estate trends, insights, current market conditions etc.\n"
        f"The user asked: '{state.user_question}'.\n\n"
        f"Using only the retrieved context:\n{state.retrieved_context}\n\n"
        f"Provide a summary of market analysis and respond to the user's question in clear language."
    )

    return {
        "market_analysis": response.content,
        "messages": [f"[market_specialist] DONE"],
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
    graph.add_node("market_specialist", market_specialist)
    graph.add_node("property_insights_specialist", property_insights_specialist)
    graph.add_node("investment_strategy_specialist", investment_strategy_specialist)
    graph.add_node("pick_response_mode", pick_response_mode)
    graph.add_node("quick_answer", quick_answer)
    graph.add_node("detailed_answer", detailed_answer)

    graph.add_edge(START, "understand_question")
    graph.add_edge("understand_question", "search_index")

    graph.add_edge("search_index", "market_specialist")
    graph.add_edge("search_index", "property_insights_specialist")
    graph.add_edge("search_index", "investment_strategy_specialist")

    graph.add_edge("market_specialist", "pick_response_mode")
    graph.add_edge("property_insights_specialist", "pick_response_mode")
    graph.add_edge("investment_strategy_specialist", "pick_response_mode")

    graph.add_conditional_edges(
        "pick_response_mode",
        route_after_decision,
        {
            "quick": "quick_answer",
            "detailed": "detailed_answer",
        },
    )

    graph.add_edge("quick_answer", END)
    graph.add_edge("detailed_answer", END)

    return graph.compile()

def query_rag(question: str) -> str:
    """Run one user question through the real estate LangGraph agent."""
    app = build_real_estate_agent()
    result = app.invoke(
        {
            "user_question": question,
            "messages": [],
        }
    )
    return result["final_answer"]

if __name__ == "__main__":
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
    answer = query_rag("Tell me how is the real estate market in 2026.")
    print(answer)
