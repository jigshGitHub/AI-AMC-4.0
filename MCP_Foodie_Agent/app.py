"""Streamlit UI for FoodieAgent v2.

Run from the project root:
    streamlit run app.py
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path for flat imports (config, llm, state, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from rag.chroma_store import seed_if_empty
from master_graph import build_master_graph

load_dotenv()

st.set_page_config(page_title="FoodieAgent v2", page_icon="🍽️", layout="wide")
st.title("🍽️ FoodieAgent v2")
st.caption(
    "Hierarchical multi-agent system  ·  Master graph + 2 compiled subgraphs  ·  "
    "Real MCP protocol (DuckDuckGo over stdio)  ·  Full guardrail stack"
)


@st.cache_resource
def _init():
    seed_if_empty()
    return build_master_graph()


graph = _init()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if user_input := st.chat_input("e.g. 'Best biryani in Mumbai under 500'  ·  'Veg South Indian in Bangalore'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("Running multi-agent pipeline…", expanded=True) as status:
            trace_lines = []
            for event in graph.stream({"user_query": user_input}):
                for node_name, node_state in event.items():
                    trace_lines.append(
                        f"**▸ {node_name}** → `{list(node_state.keys())}`"
                    )
                    st.markdown(trace_lines[-1])
            status.update(label="✓ Pipeline complete", state="complete")

        result = graph.invoke({"user_query": user_input})
        answer = result.get("final_answer") or result.get("aggregated_answer", "(no answer)")

        st.markdown("### Recommendation")
        st.markdown(answer)

        with st.expander("🧠 Full state (discovery score, RAG score, guardrail results)"):
            st.json({k: v for k, v in result.items() if k != "user_query"})

        st.session_state.messages.append({"role": "assistant", "content": answer})
