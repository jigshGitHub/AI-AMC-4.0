# Based on this  https://docs.langchain.com/oss/python/langchain/structured-output#tool-calling-strategy
# Look more examples in the docs: https://docs.langchain.com/oss/python/langchain/agents/agent-strategies#provider-strategy
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent

"""
response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]
"""
os.system('cls' if os.name=='nt' else 'clear')
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("LLM_MODEL")
agent = create_agent(
    model=model_name,
    response_format=ContactInfo  # Schema type is provided so it Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})
# 1. Print all messages
for msg in result["messages"]:
    print(f"{msg.type.upper()}: {msg.content}")

# 2. Extract the specific final AIMessage
final_ai_msg = next(m for m in reversed(result["messages"]) if m.type == "ai")
print(f"Final AI Content: {final_ai_msg.content}")

# 3. Access the validated structured data (if response_format was used)
if "structured_response" in result:
    print(f"Validated Data: {result['structured_response'].model_dump()}")
