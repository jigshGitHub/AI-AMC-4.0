"""
This resume evaluator agent uses two tools first it scans resume contents and then makes the analysis.
Look for the counter logic (syntax "if counter >") to scan how many files into the folder.
On the input prompt type C or c to use the sample JD provided in the code. You can also paste your own JD to evaluate the resumes against that.
It uses OpenAI endpoints
"""

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
from typing import List
from pydantic import BaseModel, Field

load_dotenv()
llm_model = os.getenv("LLM_MODEL")
llm = ChatOpenAI(
    model=llm_model,
    temperature=0.7,
    verbose=True,
)
class ResumeEvaluation(BaseModel):
    """Schema for the structured resume evaluation output."""
    candidate_details: str = Field(description="Full name and contact details of the candidate")
    match_score: int = Field(description="Match percentage (0-100) based on job description")
    executive_summary: str = Field(description="A 3-sentence overview of the candidate’s fitness")
    strengths: List[str] = Field(description="Bullet points of matching skills/experience")
    gaps: List[str] = Field(description="Missing skills or under-represented experience")
    recommendations: List[str] = Field(description="Actionable advice to improve the resume for this role")
    interview_verdict: str = Field(description="'Yes', 'Neutral', or 'No' with a 1-sentence justification")


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
        Contact Details including Full name, Email address and Phone Number, Core Qualifications, Hard Skills Match, Relevant Achievements, Soft Skills & Culture Fit etc.
        Attributed Perspectives: If the article contains quotes or opinions, attribute them clearly (e.g., "Source X claimed...")."""
    )

    formatted_prompt = extract_prompt.format(contents=contents)
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
    analyze_prompt = PromptTemplate(
        input_variables=["extracted_content", "job_description"],
        template="""You are a Human resource specialist for recruiting candidates in your company.
        Contents: {extracted_content}
        Job Description: {job_description}
        Task: Analyze the provided extracted content against the job descirption and generate a structured summary.
        Do not add outside information or personal interpretation.
        """
    )

    formatted_prompt = analyze_prompt.format(extracted_content=extracted_content, job_description=job_description)
    response = llm.invoke(formatted_prompt)
    return response.content

def create_resume_evaluation_agent(job_description: str):

    tools = [extract_resume_contents, analyze_resume_contents]

    SYSTEM_PROMPT ="""
                You are an expert recruiter. Your task is to extract resume contents and then evaluate its alignment with a
                provided Job Description (JD). You must provide a structured, objective summary that highlights the candidate's fitness
                for the specific role priovided in job description.
                Job Description: {job_description}
                First use extract_resume_contents then analyze_resume_contents to synthesize the extracted information into a concise evaluation
                of the candidate's suitability for the job description provided.
                Constraint:Stay strictly objective. Do not infer skills that are not explicitly stated or strongly implied by professional titles. If the JD requires "Python" and it isn't listed, mark it as a gap.
                Finally, you MUST provide your answer in the specified structured JSON format.
                """

    agent_graph = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, debug=False,response_format=ResumeEvaluation)
    return  agent_graph

def evaluate_resumes(folder_path, job_description):
    # Ensure the path exists
    if not os.path.exists(folder_path):
        logger.error("The specified folder does not exist.")
        return
    agent_graph = create_resume_evaluation_agent(job_description)

    counter = 0
    output_summary = []
    # Loop through every file in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                file_contents = ""

                logger.info(f"Evaluating file {filename}")

                if filename.endswith('.pdf'):
                    with pdfplumber.open(file_path) as pdf:
                        file_contents = "\n".join(page.extract_text() or "" for page in pdf.pages)
                elif filename.endswith('.docx'):
                    doc = docx.Document(file_path)
                    file_contents = "\n".join([para.text for para in doc.paragraphs])

                result = agent_graph.invoke(
                    {"messages": [HumanMessage(content=file_contents)]}
                )

                if "structured_response" in result:
                    evaluation = result["structured_response"]
                    # print("\n--- EVALUATION JSON ---")
                    # print(evaluation.model_dump_json(indent=2))
                    output_summary.append(evaluation)
                else:
                    output_summary.append(result["messages"][-1].content)


                logger.info(f"DONE: Evaluating file {filename}")
                #logger.info(summarized_contents)

                counter += 1
                if counter > 3: # Adjust this threshold as needed to process more or fewer files in the folder
                    break  # Remove this break to process all files in the folder
            except Exception as e:
                logger.error(f"Could not read file {filename}: {e}")

    print("\n--- FINAL EVALUATION ---")
    for evaluation in output_summary:
        print(f"Candidate Name: {evaluation.candidate_details}")
        print(f"Match Score: {evaluation.match_score}%")
        print(f"Executive Summary: {evaluation.executive_summary}")
        print("Strengths:")
        for strength in evaluation.strengths:
            print(f"- {strength}")
        print("Gaps:")
        for gap in evaluation.gaps:
            print(f"- {gap}")
        print("Recommendations:")
        for rec in evaluation.recommendations:
            print(f"- {rec}")
        print(f"Interview Verdict: {evaluation.interview_verdict}")
        print("\n-------------------------------------------------\n")

if __name__ == "__main__":
    os.system('cls' if os.name=='nt' else 'clear')
    while True:
        job_description = input("Paste your job description, to continue with sample JD, just type C or C :").strip()
        # job_description = sys.stdin.read()

        if not job_description:
            logger.error("The contents of the job description are required to move forward\n")
            continue

        if job_description.lower() in ("quit", "exit", "q"):
            logger.info("Goodbye!")
            break

        if job_description.lower() == "c":
            job_description = sample_jd

        try:
            job_description = job_description.strip()
            evaluate_resumes('./resume_files', job_description)
        except Exception as e:
            logger.error(f"\nError: {e}")
            logger.info("Please check your API key and try again.\n")
