"""
This resume evaluator agent does not use any tool but with just simple prmpt it analyze the resume.
Look for the counter logic (syntax "if counter >") to scan how many files into the folder.
On the input prompt type C or c to use the sample JD provided in the code. You can also paste your own JD to evaluate the resumes against that.
It uses OpenAI endpoints
"""
import os
import pdfplumber
import docx
import applogging
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()
logger = applogging.get_logger("resume_app")

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

def create_resume_evaluation_agent():
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=0)

    # We use a structured output LLM directly
    structured_llm = llm.with_structured_output(ResumeEvaluation)
    return structured_llm

def evaluate_resumes(folder_path, job_description):
    if not os.path.exists(folder_path):
        logger.error("Folder not found.")
        return

    evaluator = create_resume_evaluation_agent()
    output_summary = []
    counter = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_contents = ""

        try:
            if filename.endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    file_contents = "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif filename.endswith('.docx'):
                doc = docx.Document(file_path)
                file_contents = "\n".join([para.text for para in doc.paragraphs])

            if not file_contents.strip():
                continue

            # Direct invocation for structured extraction
            prompt = f"""
            You are an expert recruiter. Compare the Resume below against the Job Description.

            JOB DESCRIPTION:
            {job_description}

            RESUME:
            {file_contents}
            """

            evaluation = evaluator.invoke(prompt)
            output_summary.append(evaluation)
            logger.info(f"Evaluated {filename}")

            counter += 1
            if counter > 3:
                break

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    # Final Print Loop
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
