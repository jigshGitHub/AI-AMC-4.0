import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

class Celebrity(BaseModel):
    """Schema for the structured resume evaluation output."""
    full_details: str = Field(description="Full name and other details")
    known_for: str = Field(description="How he/she has been identified")
    executive_summary: str = Field(description="1-2-sentence overview")

load_dotenv()
llm = AzureAIOpenAIApiChatModel(
	project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
	credential = DefaultAzureCredential(exclude_environment_credential=True, exclude_managed_identity_credential=True) ,
	model = os.getenv("LLM_MODEL"),
)

parser = PydanticOutputParser(pydantic_object=Celebrity)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


structured_llm = llm.with_structured_output(Celebrity)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 4. Chain the components
    chain = prompt | llm | parser
    
    try:
        query = "Who is AB in bollywood?"
        response = chain.invoke({"query": query})
        
        print("--- CLEAN STRUCTURED OUTPUT ---")
        print(f"Name: {response.full_details}")
        print(f"Known For: {response.known_for}")
        print(f"Summary: {response.executive_summary}")
        
    except Exception as e:
        print(f"Error: {e}")

