import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

class ResumeEvaluation(BaseModel):
    """Schema for the structured resume evaluation output."""
    candidate_details: str = Field(description="Full name and contact details")
    match_score: int = Field(description="Match percentage (0-100)")
    executive_summary: str = Field(description="3-sentence overview")
    strengths: List[str] = Field(description="List of matching skills")
    gaps: List[str] = Field(description="Missing skills/experience")
    recommendations: List[str] = Field(description="Advice for the candidate")
    interview_verdict: str = Field(description="'Yes', 'No', or 'Neutral' with 1-sentence logic")

sample_jd = """
Conceives, designs, and tests logical structure to meet program requirements. Writes programs according to specifications provided. Builds, deploys and maintains programs, Web Site pages and applications. Develops and improves site navigation and applications. Responsible for the design, development, and configuration of software systems to meet market and/or client requirements. Updates, repairs, modifies, and expands existing computer programs. Writes, tests, and maintains computer programs. Develops code using Java, C#, HTML, Javascript, or other programming languages.

Responsible for design and development of Java code for a large-scale Federal IT Program.
Provides technical site maintenance and advice on moderately complex issues related to animation, search engine techniques, link integrity, navigation, browsers, graphics, and other technical web developments.
Prepares functional specifications from which programs will be written and then designs, codes, debugs, and documents programs.
Develops the requirements of a product from inception to conclusion. Tests, debugs, and refines the software to produce the required product
Designs user interfaces of interactive web applications including ADA 508, and cross browser compliance.
Maintains compliance with standards and conventions in developing programs.
Develops required specifications for simple to moderately complex programs or problems.
Conducts systems tests, monitors test results, and takes appropriate corrective action for the non-routine issues.
Creates coded unit tests and works with Testers/Information Assurance to address program and/or security findings.
Prepares required documentation, including block diagrams, logic flow charts and software program documentation.
Minimum Qualifications

Bachelor’s Degree in Computer Science, Information Systems or a related field or equivalent relevant experience.
2+ years of experience with programming or web development activities.
Active Secret Clearance
Ability to report to the client site in Annapolis Junction, MD (up to 3x a week)
"""

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

