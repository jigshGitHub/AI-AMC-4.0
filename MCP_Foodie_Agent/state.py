"""Shared state TypedDict for master graph and all subgraphs."""
from typing import TypedDict


class MasterState(TypedDict, total=False):
    # input
    user_query: str

    # input guardrail
    is_safe_input: bool
    input_rejection_reason: str

    # reformulator
    structured_query: dict  # {cuisine, location, dietary, budget, currency}

    # orchestrator decisions
    use_discovery: bool
    use_rag: bool
    orchestrator_reasoning: str

    # discovery subgraph outputs
    discovery_results: list
    discovery_score: float
    discovery_retry_count: int

    # rag subgraph outputs
    rag_context: str
    rag_score: float
    rag_answer: str
    rag_retry_count: int

    # aggregation
    aggregated_answer: str

    # output guardrail
    is_safe_output: bool
    output_warnings: list

    # tone check
    final_answer: str
