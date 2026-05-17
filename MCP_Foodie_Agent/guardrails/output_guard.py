"""Output guardrail — runs after answer aggregation, before tone check."""
import json
from llm import llm


def check_output(answer: str) -> tuple[bool, list[str], str]:
    prompt = f"""You are an output guardrail for a restaurant chatbot.
Inspect this proposed answer:

---
{answer}
---

Check for:
1. PII (real phone numbers, emails, private individuals' addresses)
2. Hallucinated specifics (fabricated restaurant names)
3. Off-topic content
4. Unsafe advice (allergy claims that could cause harm)

JSON only:
{{"is_safe": true|false, "warnings": ["..."], "cleaned_answer": "..."}}

If is_safe true, cleaned_answer = original (lightly cleaned if needed).
If is_safe false, cleaned_answer = a safe replacement message."""
    resp = llm.invoke(prompt).content.strip()
    resp = resp.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(resp)
        return (
            bool(data.get("is_safe", True)),
            list(data.get("warnings", [])),
            str(data.get("cleaned_answer", answer)),
        )
    except Exception:
        return True, [], answer
