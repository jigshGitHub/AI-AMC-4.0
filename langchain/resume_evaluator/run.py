
from asyncio.log import logger
import os
import sys
import pdfplumber
import docx
import tempfile


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

@tool
def extract_resume_contents(contents: str, job_description: str) -> str:
    """
    Use this tool FIRST when the resume content is provided.
    Extracts key information from the resume content to facilitate structured evaluation against the job description.
    and to identify sections like Core Qualifications, Hard Skills, Relevant Achievements, Soft Skills, and Culture Fit.
    """
    extract_prompt = PromptTemplate(
        input_variables=["contents", "job_description"],
        template="""You are a recruiter who can extract contents from resumes.
        Contents: {contents}
        Job Description: {job_description}
        Task: Analyze the provided resume contents to extract core information.
        Do not add outside information or personal interpretation.
        Provide extracted structured output with following
        Core Qualifications, Hard Skills Match, Relevant Achievements, Soft Skills & Culture Fit etc.
        Attributed Perspectives: If the article contains quotes or opinions, attribute them clearly (e.g., "Source X claimed...")."""
    )

    formatted_prompt = extract_prompt.format(contents=contents, job_description=job_description)
    response = llm.invoke(formatted_prompt)

    return response.content

@tool
def summarize_resume_contents(extracted_content: str, job_description: str) -> str:
    """
    Take extracted resume contents, and synthesize them into a concise evaluation of the candidate's suitability for the job description provided.
    """
    summarize_prompt = PromptTemplate(
        input_variables=["extracted_content", "job_description"],
        template="""You are a senior resume evaluator specialized in high-density factual synthesis.
        Contents: {extracted_content}
        Job Description: {job_description}
        Task: Analyze the provided extracted content and generate a structured summary. Do not add outside information or personal interpretation.

        Match Score: A percentage (0-100%) based on how well the candidate meets the "Must-Have" requirements.
        Executive Summary: A 3-sentence overview of the candidate’s profile relative to this job.
        Strengths: Bullet points of where the candidate exceeds or perfectly hits JD requirements.
        Gaps: Specific required skills or experiences that are missing or under-represented.
        Interview Verdict: A "Strong Yes," "Neutral," or "No" recommendation with a one-sentence justification.
        Constraint:
        Stay strictly objective. Do not infer skills that are not explicitly stated or strongly implied by professional titles. If the JD requires "Python" and it isn't listed, mark it as a gap.
        """
    )

    formatted_prompt = summarize_prompt.format(extracted_content=extracted_content, job_description=job_description)
    response = llm.invoke(formatted_prompt)
    return response.content



def create_resume_evaluation_agent(job_description: str):


    tools = [extract_resume_contents, summarize_resume_contents]
    SYSTEM_PROMPT ="""
                You are an expert Technical Recruiter. Your task is to analyze a user’s uploaded resume and evaluate its alignment with a
                provided Job Description (JD). You must provide a structured, objective summary that highlights the candidate's fitness for
                the specific role.
                Job Description: {job_description}
                First use extract_resume_contents to extract Core Qualifications, Hard Skills Match, Relevant Achievements, Soft Skills & Culture Fit etc.
                Then use summarize_resume_contents to synthesize the extracted information into a concise evaluation of the candidate's suitability for the job description provided.
                Constraint: Stay strictly objective. Do not infer skills that are not explicitly stated or strongly implied by professional titles. If the JD requires "Python" and it isn't listed, mark it as a gap.
                Use Output Format as below:
                Match Score: A percentage (0-100%) based on how well the candidate meets the "Must-Have" requirements.
                Executive Summary: A 3-sentence overview of the candidate’s profile relative to this role.
                Strengths: Bullet points of where the candidate exceeds or perfectly hits JD requirements.
                Gaps: Specific required skills or experiences that are missing or under-represented.
                Interview Verdict: A "Strong Yes," "Neutral," or "No" recommendation with a one-sentence justification.
                """

    agent_graph = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, debug=True)
    return  agent_graph

def evaluate_resumes(folder_path, job_description):
    # Ensure the path exists
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return
    agent_graph = create_resume_evaluation_agent(job_description)
    # Loop through every file in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                file_contents = ""

                if filename.endswith('.pdf'):
                    print(f"Reading pdf file {filename}")
                    with pdfplumber.open(file_path) as pdf:
                        file_contents = "\n".join(page.extract_text() or "" for page in pdf.pages)
                elif filename.endswith('.docx'):
                    print(f"Reading Eord document file {filename}")
                    doc = docx.Document(file_path)
                    file_contents = "\n".join([para.text for para in doc.paragraphs])

                result = agent_graph.invoke(
                    {"messages": [HumanMessage(content=file_contents)]}
                )

                summarized_contents = result["messages"][-1].content
                print(summarized_contents)
                print(f"=" * 60)
                break  # Remove this break to process all files in the folder
            except Exception as e:
                print(f"Could not read file {filename}: {e}")

if __name__ == "__main__":
    while True:
        # news_contents = input("Please paste the contents of the news article: ")

        print("Paste your job description and enter Ctrl+Z then enter to finish:")
        job_description = sys.stdin.read()
        job_description = job_description.strip()

        if not job_description:
            print("The contents of the job description are required to move forward\n")
            continue

        if job_description.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        try:
            evaluate_resumes('./resume_files', job_description)
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your API key and try again.\n")
