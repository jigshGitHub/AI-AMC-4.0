# Based on this  https://docs.langchain.com/oss/python/langchain/structured-output#tool-calling-strategy
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent


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
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')