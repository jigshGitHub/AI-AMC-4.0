import os
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()
def get_azOpenAIClient():
    if os.getenv("AZURE_CREDENTIAL") == 'default':
        print('default  needs development')
    else:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint= os.getenv("AZURE_RESOURCE_ENDPOINT")
        )
        return client

def get_openAIClient():
    if os.getenv("AZURE_CREDENTIAL") == 'default':
        print('default  needs development')
    else:
        client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        return client

