"""RAG subgraph: agentic guardrail → retrieval → search eval → answer eval."""
import json
from langgraph.graph import StateGraph, END

from state import MasterState
from guardrails.agentic_guard import check_for_rag
from rag.chroma_store import retrieve
from llm import llm
from config import (
    SEARCH_EVAL_THRESHOLD,
    MAX_RETRIES_PER_SUBGRAPH,
    RAG_TOP_K,
)


def agentic_guardrail_node(state: MasterState) -> dict:
    sq = state.get("structured_query", {})
    is_safe, reason = check_for_rag(sq)
    if not is_safe:
        return {
            "rag_context": "",
            "rag_score": 0.0,
            "rag_answer": f"(rag blocked: {reason})",
        }
    # Guardrail passed — return a no-op update (LangGraph requires at least one key)
    return {"rag_retry_count": state.get("rag_retry_count", 0)}


def retrieval_node(state: MasterState) -> dict:
    if state.get("rag_answer", "").startswith("(rag blocked"):
        return {"rag_context": ""}  # blocked — propagate empty context forward
    sq = state.get("structured_query", {})
    cuisine = sq.get("cuisine") or ""
    dietary = sq.get("dietary") or ""
    query = f"{cuisine} {dietary}".strip() or state.get("user_query", "")
    chunks = retrieve(query, k=RAG_TOP_K)
    return {"rag_context": "\n\n---\n\n".join(chunks)}


def search_evaluation_node(state: MasterState) -> dict:
    ctx = state.get("rag_context", "")
    if not ctx:
        return {"rag_score": 0.0}
    prompt = f"""Score the relevance (0.0 to 1.0) of these retrieved knowledge-base
chunks to the user's restaurant intent.

User query: {state.get("user_query", "")}
Chunks (first 2000 chars): {ctx[:2000]}

JSON only: {{"score": 0.0-1.0, "reason": "..."}}"""
    resp = llm.invoke(prompt).content.strip()
    resp = resp.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(resp)
        return {"rag_score": float(data.get("score", 0.5))}
    except Exception:
        return {"rag_score": 0.5}


def answer_evaluation_node(state: MasterState) -> dict:
    ctx = state.get("rag_context", "")
    if not ctx:
        return {"rag_answer": "(no relevant cuisine info found)"}
    prompt = f"""Using ONLY the context below, write 2-3 sentences covering the
cuisine and dietary aspects relevant to the user's question.
If the context is insufficient, say so explicitly.

Context:
{ctx}

User question: {state.get("user_query", "")}

Answer:"""
    return {"rag_answer": llm.invoke(prompt).content.strip()}


def _route_after_search_eval(state: MasterState) -> str:
    score = state.get("rag_score", 0.0)
    retries = state.get("rag_retry_count", 0)
    if score < SEARCH_EVAL_THRESHOLD and retries < MAX_RETRIES_PER_SUBGRAPH:
        return "retry"
    return "answer_eval"


def retry_prep_node(state: MasterState) -> dict:
    return {
        "rag_retry_count": state.get("rag_retry_count", 0) + 1,
        "rag_context": "",
    }


def build_rag_subgraph():
    sg = StateGraph(MasterState)
    sg.add_node("agentic_guardrail", agentic_guardrail_node)
    sg.add_node("retrieval", retrieval_node)
    sg.add_node("search_evaluation", search_evaluation_node)
    sg.add_node("answer_evaluation", answer_evaluation_node)
    sg.add_node("retry_prep", retry_prep_node)

    sg.set_entry_point("agentic_guardrail")
    sg.add_edge("agentic_guardrail", "retrieval")
    sg.add_edge("retrieval", "search_evaluation")
    sg.add_conditional_edges(
        "search_evaluation",
        _route_after_search_eval,
        {"retry": "retry_prep", "answer_eval": "answer_evaluation"},
    )
    sg.add_edge("retry_prep", "retrieval")
    sg.add_edge("answer_evaluation", END)

    return sg.compile()
