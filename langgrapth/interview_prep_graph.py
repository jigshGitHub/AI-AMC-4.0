import operator
import sys
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

class InterviewState(BaseModel):
    job_role:str = ""
    urgency_level : str = ""
    technical_suggestion : str = ""
    behavioral_suggestion : str = ""
    confidence_suggestion : str = ""
    needs_deep_preparation: bool = False
    practice_reason: str = ""
    final_suggestion: str = ""
    messages:Annotated[list, operator.add] = []

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def understand_candidate(state:InterviewState):
    prompt = f"""
    You are an expert interview coach. Based on candidate preparing an interview for the position as {state.job_role}, first
    acknowledge the candidate's preparation and classify the urgency level as LOW, MEDIUM, or HIGH
    in one word on a new line like: Urgency LEVEL: MEDIUM
    )"""
    response = llm.invoke(prompt)
    return {
        "messages": [f"[understand_candidate] {response.content}"]
    }

def suggest_technical(state:InterviewState):
    prompt = f"""
    You are an expert interview coach. Based on the candidate's preparation for the position as {state.job_role}, suggest some specific technical topics or skills that the candidate should focus on to improve their chances of success in the interview. Provide a brief explanation of why these topics are important for the role.
    """
    response = llm.invoke(prompt)
    return {
        "technical_suggestion": response.content,
        "messages": [f"[suggest_technical] Done"]
    }

def suggest_behavioral(state:InterviewState):
    prompt = f"""
    You are an expert interview coach. Based on the candidate's preparation for the position as {state.job_role}, suggest one specific behavioral aspect or soft skill that the candidate should focus on to improve their chances of success in the interview. Provide a brief explanation of why this aspect is important for the role.
    """
    response = llm.invoke(prompt)
    return {
        "behavioral_suggestion": response.content,
        "messages": [f"[suggest_behavioral] Done"]
    }

def suggest_confidence(state:InterviewState):
    prompt = f"""
    You are an expert interview coach. Based on the candidate's preparation for the position as {state.job_role}, suggest one specific confidence-building technique or mindset shift that the candidate should focus on to improve their chances of success in the interview. Provide a brief explanation of why this technique or mindset is important for the role.
    """
    response = llm.invoke(prompt)
    return {
        "confidence_suggestion": response.content,
        "messages": [f"[suggest_confidence] Done"]
    }

def pick_best_practice(state:InterviewState):
    prompt = f"""
    You are an expert interview coach.
    Based on your preparation for the {state.job_role} role, here are personalized
    suggestions from the experts:
    1. Technical Focus: {state.technical_suggestion}
    2. Behavioral Focus: {state.behavioral_suggestion}
    3. Confidence Building: {state.confidence_suggestion}

    Decide: does this candidate need a QUICK PREPARATION (1-hour or so for focused drill) "
    or a DEEP preparation that includes 3-4 hours of structured study plan and practice?
    Reply STRICTLY in this JSON format (no other text):
    {{"needs_deep_preparation": true/false, "reason": "one sentence explanation"}}'
    """
    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
        needs_deep = result["needs_deep_preparation"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_deep = False
        reason = "Could not parse decision, defaulting to quick practice."

    return {
        "needs_deep_preparation": needs_deep,
        "practice_reason": reason,
        "messages": [f"[pick_best_practice] deep_session={needs_deep}"]
    }

def route_after_decision(state: InterviewState) -> str:
    if state.needs_deep_preparation:
        return "deep"
    else:
        return "quick"

def quick_practice(state: InterviewState) -> dict:
    response = llm.invoke(
        f"You are an interview preparation coach. The candidate is preparing for the position: '{state.job_role}'.\n\n"
        f"Based on these specialist suggestions, create a QUICK structured preparation approach (1 hour or so) "
        f"that thoughtfully combines all three suggestions:\n\n"
        f"TECHNICAL TOPICS: {state.technical_suggestion}\n"
        f"BEHAVIORAL TOPICS: {state.behavioral_suggestion}\n"
        f"CONFIDENCE BUILDING: {state.confidence_suggestion}\n"
        f"Structure all suggestions in 3 phases: Technical Approach (technical), Behavioral Approach (behavioral) and Confidence Building (confidence). "
        f"Give clear step-by-step instructions for each phase with timing. "
        f"Keep it warm and supportive. End with a kind closing message."
    )
    return {
        "final_suggestion": f"DEEP PREPARATION TACTICS (1-2 hours)\n{'='*45}\n{response.content}",
        "messages": [f"[deep_practice] Generated deep approach"]
    }

def deep_practice(state: InterviewState) -> dict:
    response = llm.invoke(
        f"You are an interview preparation coach. The candidate is preparing for the position: '{state.job_role}'.\n\n"
        f"Based on these specialist suggestions, create a DEEPER structured preparation approach (5-6 hours) "
        f"that thoughtfully combines all three suggestions:\n\n"
        f"TECHNICAL TOPICS: {state.technical_suggestion}\n"
        f"BEHAVIORAL TOPICS: {state.behavioral_suggestion}\n"
        f"CONFIDENCE BUILDING: {state.confidence_suggestion}\n"
        f"Structure all suggestions in 3 phases: Technical Approach (technical), Behavioral Approach (behavioral) and Confidence Building (confidence). "
        f"Give clear step-by-step instructions for each phase with timing. "
        f"Keep it warm and supportive. End with a kind closing message."
    )
    return {
        "final_suggestion": f"DEEP PREPARATION TACTICS (5-6 hours)\n{'='*45}\n{response.content}",
        "messages": [f"[deep_practice] Generated deep approach"]
    }


graph = StateGraph(InterviewState)

graph.add_node("understand_candidate", understand_candidate)
graph.add_node("suggest_technical", suggest_technical)
graph.add_node("suggest_behavioral", suggest_behavioral)
graph.add_node("suggest_confidence", suggest_confidence)
graph.add_node("pick_best_practice", pick_best_practice)
graph.add_node("quick_practice", quick_practice)
graph.add_node("deep_practice", deep_practice)

graph.add_edge(START, "understand_candidate")

graph.add_edge("understand_candidate", "suggest_technical")
graph.add_edge("understand_candidate", "suggest_behavioral")
graph.add_edge("understand_candidate", "suggest_confidence")

graph.add_edge("suggest_technical", "pick_best_practice")
graph.add_edge("suggest_behavioral", "pick_best_practice")
graph.add_edge("suggest_confidence", "pick_best_practice")

graph.add_conditional_edges(
    "pick_best_practice",
    route_after_decision,
    {
        "quick": "quick_practice",
        "deep": "deep_practice",
    }
)

graph.add_edge("quick_practice", END)
graph.add_edge("deep_practice", END)
graph.add_edge("pick_best_practice", END)
app = graph.compile()

def run_interview_preparation(job_role:str):
    print("=" * 55)
    print("  INTERVIEW PREPARATION SUGGESTER")
    print(f"  You said: \"{job_role}\"")
    print("=" * 55)

    result = app.invoke({
        "job_role": job_role,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print("  YOUR PERSONALIZED PRACTICE")
    print("=" * 55)
    print(f"\n{result['final_suggestion']}")


    print("\n" + "-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  INTERVIEW PREP COACH")
    print("=" * 55)
    print("\n  Tell me what kind of role are you preparing for and how is your preparation going? ")
    print("  I'll provide a personalized interview preparation strategy just for you.")
    print("  Type 'quit' to exit.\n")

    while True:
        role = input(" So now tell me what kind of role are you preparing for and how is your preparation going? > ").strip()

        if role.lower() in ("quit", "exit", "q"):
            print("\n  Take care of yourself. Goodbye!\n")
            break

        if not role:
            continue

        run_interview_preparation(role)
        print("\n")
