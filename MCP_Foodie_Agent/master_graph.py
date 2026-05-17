"""Master graph wiring. Subgraphs are compiled and added as nodes."""
from langgraph.graph import StateGraph, END

from state import MasterState
from master_nodes import (
    input_guardrail_node,
    reformulator_node,
    orchestrator_node,
    reject_node,
    answer_aggregator_node,
    output_guardrail_node,
    tone_node,
)
from subgraphs.discovery_subgraph import build_discovery_subgraph
from subgraphs.rag_subgraph import build_rag_subgraph


def _route_after_input_guard(state: MasterState) -> str:
    return "reformulator" if state.get("is_safe_input") else "reject"


def build_master_graph():
    g = StateGraph(MasterState)

    # Master nodes
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("reformulator", reformulator_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("reject", reject_node)
    g.add_node("aggregator", answer_aggregator_node)
    g.add_node("output_guardrail", output_guardrail_node)
    g.add_node("tone", tone_node)

    # Compiled subgraphs as nodes
    g.add_node("discovery_subgraph", build_discovery_subgraph())
    g.add_node("rag_subgraph", build_rag_subgraph())

    # Wiring — sequential execution (parallel fan-out requires LangGraph >=0.3)
    g.set_entry_point("input_guardrail")
    g.add_conditional_edges(
        "input_guardrail",
        _route_after_input_guard,
        {"reformulator": "reformulator", "reject": "reject"},
    )
    g.add_edge("reject", END)
    g.add_edge("reformulator", "orchestrator")
    g.add_edge("orchestrator", "discovery_subgraph")
    g.add_edge("discovery_subgraph", "rag_subgraph")
    g.add_edge("rag_subgraph", "aggregator")
    g.add_edge("aggregator", "output_guardrail")
    g.add_edge("output_guardrail", "tone")
    g.add_edge("tone", END)

    return g.compile()
