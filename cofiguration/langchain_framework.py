import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

def get_llm():
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "azure":
        if os.getenv("AZURE_CREDENTIAL") == 'default':
            llm = AzureAIOpenAIApiChatModel(
                project_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                credential = DefaultAzureCredential(exclude_environment_credential=True, exclude_managed_identity_credential=True) ,
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                temperature=0.7,
                verbose=True,
            )
        else:
            llm = AzureAIOpenAIApiChatModel(
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                credential=os.getenv("AZURE_OPENAI_API_KEY"),
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                temperature=0.7,
                verbose=True,
            )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-proj"):
            raise ValueError("OPENAI_API_KEY not set! Copy .env.example to .env and add your key.")
            sys.exit(1)
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=0.7,
            verbose=True,
        )

    return llm

def get_agent(tools=None, system_prompt=None,debug=False,response_format=None):
    """
    Creates and returns an agent using the configured LLM.

    tools: LangChain tools available to the agent.
    system_prompt: Optional instructions for the agent.
    """
    llm = get_llm()

    if llm is None:
        raise RuntimeError(
            "get_llm() returned None. Check LLM_PROVIDER and ensure every "
            "provider branch in get_llm() returns a valid chat-model instance."
        )

    return create_agent(
        model=llm,
        tools=tools or [],
        system_prompt=system_prompt or "You are a helpful assistant.",
        debug=debug,
        response_format=response_format
    )

def get_embedding_llm():
    """Return an embedding model instance connected to Azure OpenAI."""

    # Securely retrieve endpoints and keys from your environment variables
    azure_endpoint = os.getenv("AZURE_RESOURCE_ENDPOINT")#, "https://azure.com")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")#, "your-actual-azure-api-key")

    print("Returning AzureOpenAIEmbeddings")
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("TEXT_EMBEDDING_MODEL"),      # The name of your model deployment in Azure
        azure_endpoint=azure_endpoint,    # The base URL (LangChain automatically appends /openai/v1)
        api_key=api_key,                  # Your project API key
        api_version="2024-02-01"          # Standard stable Azure API version
    )
