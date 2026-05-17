"""Tone-of-voice enforcement — final node before output."""
from llm import llm


def enforce_tone(answer: str) -> str:
    prompt = f"""Rewrite this restaurant chatbot answer in a friendly, warm,
professional tone. Keep all factual content identical. Preserve markdown.
Do not add new facts or remove restaurant names, citations, or URLs.

Original:
---
{answer}
---

Rewritten answer:"""
    return llm.invoke(prompt).content.strip()
