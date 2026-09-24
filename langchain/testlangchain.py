
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from azure.identity import DefaultAzureCredential
import os

def create_my_custom_agent(model_name: str = "openai:gpt-4o"):
    load_dotenv()
    env_type = os.getenv("ENVTYPE")
    if env_type == "azure":
        from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
        if os.getenv("AZURE_CREDENTIAL") == 'default':
            llm = AzureAIOpenAIApiChatModel(
                project_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                credential = DefaultAzureCredential(exclude_environment_credential=True, exclude_managed_identity_credential=True) ,
	            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0.7,
                verbose=True,
            )
        else:
            # key credential
            llm = AzureAIOpenAIApiChatModel(
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                credential=os.getenv("AZURE_OPENAI_API_KEY"),
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0.7,
                verbose=True,
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
        system_prompt="You are a helpful assistant."
    )
    return agent


if __name__ == "__main__":
    my_agent = create_my_custom_agent()
    #inputs = {"messages": [{"role": "user", "content": "What's the weather in Washington DC?IS it good time to visit this year"}]}
    inputs = {"messages": [{"role": "user", "content": "Who is AB in Bollywood?"}]}

    # for chunk in my_agent.stream(inputs, stream_mode="updates"):
    #     print(chunk)

    print(my_agent.invoke(inputs)["messages"][-1].content_blocks)
