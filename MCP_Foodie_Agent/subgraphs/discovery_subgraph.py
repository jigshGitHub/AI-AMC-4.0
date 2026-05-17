"""Discovery subgraph: agentic guardrail → MCP tool exec → search eval (with retry)."""
import json
from langgraph.graph import StateGraph, END

from state import MasterState
from guardrails.agentic_guard import check_for_discovery
from mcp_clients.duckduckgo_client import search as mcp_search
from llm import llm
from config import SEARCH_EVAL_THRESHOLD, MAX_RETRIES_PER_SUBGRAPH


def agentic_guardrail_node(state: MasterState) -> dict:
    sq = state.get("structured_query", {})
    is_safe, reason = check_for_discovery(sq)
    if not is_safe:
        return {
            "discovery_results": [{"error": f"agentic guardrail blocked: {reason}"}],
            "discovery_score": 0.0,
        }
    # Guardrail passed — return a no-op update (LangGraph requires at least one key)
    return {"discovery_retry_count": state.get("discovery_retry_count", 0)}


def mcp_tool_execution_node(state: MasterState) -> dict:
    results = state.get("discovery_results", [])
    if results and "error" in results[0]:
        return {"discovery_results": results}  # propagate the blocked error forward
    sq = state.get("structured_query", {})
    cuisine = sq.get("cuisine") or "restaurant"
    location = sq.get("location") or "Mumbai"
    retry = state.get("discovery_retry_count", 0)
    # On retry, broaden the query slightly
    query = (
        f"best {cuisine} restaurants {location} reviews 2025"
        if retry == 0
        else f"top {cuisine} food places {location} recommended"
    )
    try:
        results = mcp_search(query, max_results=5)
    except Exception as e:
        results = [{"error": f"MCP call failed: {e}"}]
    return {"discovery_results": results}


def search_evaluation_node(state: MasterState) -> dict:
    results = state.get("discovery_results", [])
    if not results or (len(results) == 1 and "error" in results[0]):
        return {"discovery_score": 0.0}
    sq = state.get("structured_query", {})
    prompt = f"""Score the relevance of these web search results (0.0 to 1.0)
for the user's restaurant intent.

User intent: {json.dumps(sq)}
Results (first 2000 chars): {json.dumps(results)[:2000]}

JSON only: {{"score": 0.0-1.0, "reason": "..."}}"""
    resp = llm.invoke(prompt).content.strip()
    resp = resp.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(resp)
        return {"discovery_score": float(data.get("score", 0.5))}
    except Exception:
        return {"discovery_score": 0.5}


def _route_after_eval(state: MasterState) -> str:
    score = state.get("discovery_score", 0.0)
    retries = state.get("discovery_retry_count", 0)
    if score < SEARCH_EVAL_THRESHOLD and retries < MAX_RETRIES_PER_SUBGRAPH:
        return "retry"
    return "exit"


def retry_prep_node(state: MasterState) -> dict:
    return {
        "discovery_retry_count": state.get("discovery_retry_count", 0) + 1,
        "discovery_results": [],
    }


def build_discovery_subgraph():
    sg = StateGraph(MasterState)
    sg.add_node("agentic_guardrail", agentic_guardrail_node)
    sg.add_node("mcp_tool_execution", mcp_tool_execution_node)
    sg.add_node("search_evaluation", search_evaluation_node)
    sg.add_node("retry_prep", retry_prep_node)

    sg.set_entry_point("agentic_guardrail")
    sg.add_edge("agentic_guardrail", "mcp_tool_execution")
    sg.add_edge("mcp_tool_execution", "search_evaluation")
    sg.add_conditional_edges(
        "search_evaluation",
        _route_after_eval,
        {"retry": "retry_prep", "exit": END},
    )
    sg.add_edge("retry_prep", "mcp_tool_execution")

    return sg.compile()
