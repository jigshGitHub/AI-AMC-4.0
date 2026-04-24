from __future__ import annotations

import json
import operator
import os
import config
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE)

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

def understand_question(state: HealthFitnessState) -> dict:
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
    # graph.add_node("search_index", search_index)
    # graph.add_node("health_specialist", health_specialist)
    # graph.add_node("gym_specialist", gym_specialist)
    # graph.add_node("fitness_specialist", fitness_specialist)
    # graph.add_node("pick_response_mode", pick_response_mode)
    # graph.add_node("quick_answer", quick_answer)
    # graph.add_node("detailed_answer", detailed_answer)

    graph.add_edge(START, "understand_question")
    # graph.add_edge("understand_question", "search_index")

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

    graph.add_edge("understand_question", END)

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