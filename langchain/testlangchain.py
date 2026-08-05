
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

import os

@tool
def get_weather() -> str:
    """
    Returns the weather forecast for a given location.
    In a real implementation, this would call a weather API. For this example, it returns a static response.
    """
    return f"The weather is 72°F and sunny."

def create_my_custom_agent(model_name: str = "openai:gpt-4o"):
    load_dotenv()
    tools = [get_weather]
    env_type = os.getenv("ENVTYPE")
    if env_type == "azure":
        from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
        llm = AzureAIOpenAIApiChatModel(
            project_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            credential=None,  # Assuming DefaultAzureCredential is set up in the environment
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.7,
            verbose=True,
        )
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="You are a helpful assistant"
        )
    else:
        model_name = os.getenv("LLM_MODEL")
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            verbose=True,
        )
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="You are a helpful and concise weather assistant."
        )
    return agent


if __name__ == "__main__":
    my_agent = create_my_custom_agent()
    inputs = {"messages": [{"role": "user", "content": "What's the weather in Washington DC?"}]}

    # for chunk in my_agent.stream(inputs, stream_mode="updates"):
    #     print(chunk)

    print(my_agent.invoke(inputs)["messages"][-1].content_blocks)
