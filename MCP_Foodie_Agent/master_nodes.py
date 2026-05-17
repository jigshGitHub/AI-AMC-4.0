"""Master-level node functions: input guard, reformulator, orchestrator,
aggregator, output guard, tone check, reject."""
import json
from state import MasterState
from llm import llm
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output
from guardrails.tone_check import enforce_tone


def input_guardrail_node(state: MasterState) -> dict:
    is_safe, reason = check_input(state["user_query"])
    return {"is_safe_input": is_safe, "input_rejection_reason": reason}


def reformulator_node(state: MasterState) -> dict:
    prompt = f"""Extract structured intent from this restaurant query.

Query: "{state['user_query']}"

JSON only:
{{"cuisine": "...", "location": "...", "dietary": "veg|non-veg|vegan|null",
  "budget_max": <number or null>, "currency": "INR|USD|SGD|null"}}"""
    resp = llm.invoke(prompt).content.strip()
    resp = resp.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        sq = json.loads(resp)
    except Exception:
        sq = {"cuisine": None, "location": None, "dietary": None,
              "budget_max": None, "currency": None}
    return {"structured_query": sq}


def orchestrator_node(state: MasterState) -> dict:
    sq = state.get("structured_query", {})
    prompt = f"""You orchestrate two subagents:
- DISCOVERY (web search via MCP) — finds real restaurant candidates
- RAG (cuisine knowledge base) — provides cuisine/dietary context

Structured query: {json.dumps(sq)}
Raw user message: "{state.get('user_query', '')}"

Almost always call BOTH. Only skip RAG if there's no cuisine/dietary content.
Only skip DISCOVERY if the query is purely educational ("what is biryani?").

JSON only: {{"use_discovery": true|false, "use_rag": true|false, "reasoning": "..."}}"""
    resp = llm.invoke(prompt).content.strip()
    resp = resp.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(resp)
        return {
            "use_discovery": bool(data.get("use_discovery", True)),
            "use_rag": bool(data.get("use_rag", True)),
            "orchestrator_reasoning": str(data.get("reasoning", "")),
        }
    except Exception:
        return {"use_discovery": True, "use_rag": True, "orchestrator_reasoning": "default both"}


def reject_node(state: MasterState) -> dict:
    return {
        "final_answer": (
            "I can only help with restaurant and food questions. "
            f"({state.get('input_rejection_reason', 'off-topic')})"
        )
    }


def answer_aggregator_node(state: MasterState) -> dict:
    discovery = state.get("discovery_results", [])
    rag_answer = state.get("rag_answer", "")
    rag_context = state.get("rag_context", "")
    sq = state.get("structured_query", {})

    prompt = f"""Aggregate subagent outputs into a single restaurant recommendation.

User's query: "{state.get('user_query', '')}"
Structured intent: {json.dumps(sq)}

[DISCOVERY — live web search via MCP]
{json.dumps(discovery)[:3000]}

[RAG — cuisine knowledge base]
Grounded answer: {rag_answer}
Source chunks: {rag_context[:1500]}

Write a recommendation that:
1. Names 2-3 restaurant candidates from discovery results (use actual names/URLs from snippets)
2. Adds a sentence of cuisine/dietary context from RAG
3. Notes if either subagent returned weak results
4. Uses markdown for clarity

Do not invent restaurants not present in the discovery results."""
    return {"aggregated_answer": llm.invoke(prompt).content.strip()}


def output_guardrail_node(state: MasterState) -> dict:
    answer = state.get("aggregated_answer", "")
    is_safe, warnings, cleaned = check_output(answer)
    return {
        "is_safe_output": is_safe,
        "output_warnings": warnings,
        "aggregated_answer": cleaned,
    }


def tone_node(state: MasterState) -> dict:
    return {"final_answer": enforce_tone(state.get("aggregated_answer", ""))}
