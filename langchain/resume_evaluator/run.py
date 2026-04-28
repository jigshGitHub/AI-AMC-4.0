
from asyncio.log import logger
import os
import sys
import pdfplumber
import docx
import tempfile
import applogging


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()
llm_model = os.getenv("LLM_MODEL")
llm = ChatOpenAI(
    model=llm_model,
    temperature=0.7,
    verbose=True,
)

logger = applogging.get_logger("resume_app")

@tool
def extract_resume_contents(contents: str) -> str:
    """
    Use this tool FIRST when the resume content is provided.
    Extracts key information from the resume content to facilitate structured evaluation and to identify sections like Contact details, 
    Core Qualifications, Hard Skills, Relevant Achievements, Soft Skills, and Culture Fit.
    """
    extract_prompt = PromptTemplate(
        input_variables=["contents"],
        template="""You are a recruiter who can extract contents from resumes.
        Contents: {contents}
        Task: Analyze the provided resume contents to extract core information.
        Do not add outside information or personal interpretation.
        Provide extracted structured output with following
        Contact Details, Core Qualifications, Hard Skills Match, Relevant Achievements, Soft Skills & Culture Fit etc.
        Attributed Perspectives: If the article contains quotes or opinions, attribute them clearly (e.g., "Source X claimed...")."""
    )

    formatted_prompt = extract_prompt.format(contents=contents, job_description=job_description)
    response = llm.invoke(formatted_prompt)

    return response.content

@tool
def analyze_resume_contents(extracted_content: str, job_description: str) -> str:
    """
    Take extracted resume contents and job description, use them to synthesize into a concise evaluation of the candidate's suitability for the job description provided.
    Args:
        extracted_content: Extracted contents of resume.
        job_description: Job description to evalate against the resume contents.
    """
    
    logger.info(f"[Tool: extract_resume_contents] Received JD: '{job_description}'")

    analyze_prompt = PromptTemplate(
        input_variables=["extracted_content", "job_description"],
        template="""You are a Human resource specialist for recruiting candidates in your company.
        Contents: {extracted_content}
        Job Description: {job_description}
        Task: Analyze the provided extracted content against the job descirption provide and generate a structured summary. 
        Do not add outside information or personal interpretation.
        Provide the analysis in following out put format:
        Match Score: A percentage (0-100%) based on how well the candidate meets the "Must-Have" requirements.
        Executive Summary: A 3-sentence overview of the candidate’s profile relative to this job.
        Strengths: Bullet points of where the candidate exceeds or perfectly hits JD requirements.
        Gaps: Specific required skills or experiences that are missing or under-represented.
        Interview Verdict: A "Strong Yes," "Neutral," or "No" recommendation with a one-sentence justification.
        Constraint:
        Stay strictly objective. Do not infer skills that are not explicitly stated or strongly implied by professional titles. If the JD requires "Python" and it isn't listed, mark it as a gap.
        """
    )

    formatted_prompt = analyze_prompt.format(extracted_content=extracted_content, job_description=job_description)
    response = llm.invoke(formatted_prompt)
    return response.content



def create_resume_evaluation_agent(job_description: str):

    tools = [extract_resume_contents, analyze_resume_contents]
    
    # SYSTEM_PROMPT ="""
    #             You are an expert recruiter. Your task is to extract resume contents. As a recruiter you have following job descripton \n
    #             JobDescription: {job_description} 
    #             Use extract_resume_contents to extract Core Qualifications, Hard Skills Match, Relevant Achievements, Soft Skills & Culture Fit etc.
    #             Constraint: Stay strictly objective. Do not infer skills that are not explicitly stated or strongly implied by professional titles. 
    #             """
    SYSTEM_PROMPT ="""
                You are an expert recruiter. Your task is to extract resume contents and then evaluate its alignment with a
                provided Job Description (JD). You must provide a structured, objective summary that highlights the candidate's fitness 
                for the specific role priovided in job description.
                Job Description: {job_description}
                First use extract_resume_contents to extract Contact details, Core Qualifications, Hard Skills Match, Relevant Achievements, Soft Skills & Culture Fit etc.
                Then use analyze_resume_contents to synthesize the extracted information into a concise evaluation of the candidate's suitability for the job description provided.
                Constraint: Stay strictly objective. Do not infer skills that are not explicitly stated or strongly implied by professional titles. If the JD requires "Python" and it isn't listed, mark it as a gap.
                """

    agent_graph = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, debug=True)
    return  agent_graph

def evaluate_resumes(folder_path, job_description):
    # Ensure the path exists
    if not os.path.exists(folder_path):
        logger.error("The specified folder does not exist.")
        return
    agent_graph = create_resume_evaluation_agent(job_description)
    # Loop through every file in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                file_contents = ""

                if filename.endswith('.pdf'):
                    logger.info(f"Reading pdf file {filename}")
                    with pdfplumber.open(file_path) as pdf:
                        file_contents = "\n".join(page.extract_text() or "" for page in pdf.pages)
                elif filename.endswith('.docx'):
                    logger.info(f"Reading word document file {filename}")
                    doc = docx.Document(file_path)
                    file_contents = "\n".join([para.text for para in doc.paragraphs])

                result = agent_graph.invoke(
                    {"messages": [HumanMessage(content=file_contents)]}
                )

                summarized_contents = result["messages"][-1].content
                logger.info(summarized_contents)
                print(f"=" * 60)
                break  # Remove this break to process all files in the folder
            except Exception as e:
                logger.error(f"Could not read file {filename}: {e}")

if __name__ == "__main__":    
    os.system('cls' if os.name=='nt' else 'clear')
    while True:
        job_description = input("Paste your job description :").strip()
        # job_description = sys.stdin.read()

        if not job_description:
            logger.error("The contents of the job description are required to move forward\n")
            continue

        if job_description.lower() in ("quit", "exit", "q"):
            logger.info("Goodbye!")
            break

        try:
            job_description = """
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
            job_description = job_description.strip()
            evaluate_resumes('./resume_files', job_description)
        except Exception as e:
            logger.error(f"\nError: {e}")
            logger.info("Please check your API key and try again.\n")
