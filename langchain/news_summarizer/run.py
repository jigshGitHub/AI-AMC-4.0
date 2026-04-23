import logging
import sys
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

os.system('cls' if os.name=='nt' else 'clear')
# setup log information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("News Summarizer")
logger.info("Starting News Summarizer Agent...")

# load environment and varify
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    logger.error("OPENAI_API_KEY not set! Copy .env.example to .env and add your key.")
    sys.exit(1)
logger.info("API key loaded successfully")

gpt_model = os.getenv("GPT_MODEL")
if not gpt_model or gpt_model.startswith("gpt*"):
    logger.error("gpt model not set! Copy .env.example to .env and set GPT_MODEL.")
    sys.exit(1)
logger.info("GPT Model retrieved successfully")

logger.info("All LangChain components imported")

logger.info("Initializing the LLM (OpenAI GPT)...")
llm = ChatOpenAI(
    model=gpt_model,
    temperature=0.7,
    verbose=True,
)
logger.info(f"LLM initialized: model={gpt_model}, temperature=0.7")

logger.info("Defining agent tools...")
@tool
def extract_news_contents(contents: str) -> str:
    """
    Extract factual points,3-5 bullet points of verifiable facts (Who, What, Where, When, Why).
    Use this tool FIRST when the user provides a news article contents.
    Input should be the user's email contents or topic.
    Returns extracted contents.
    """
    logger.info(f"[Tool: extract_news_contents] Received contents: '{contents}'")

    extract_prompt = PromptTemplate(
        input_variables=["contents"],
        template="""You are an objective, neutral news analyst specializing in factual synthesis.
        Contents: {contents}
        Task: Analyze the provided news article contents to extract core information. Do not add outside information or personal interpretation.
        Provide extracted structured output with following
        Key Factual Points: 3-5 bullet points of verifiable facts (Who, What, Where, When, Why).
        Key Stakeholders: List the primary organizations, public figures, or groups involved.
        Important Data: Any specific numbers, percentages, dates, or financial figures.
        Attributed Perspectives: If the article contains quotes or opinions, attribute them clearly (e.g., "Source X claimed...")."""
    )

    formatted_prompt = extract_prompt.format(contents=contents)
    logger.info("[Tool: extract_news_contents] Sending prompt to LLM...")

    response = llm.invoke(formatted_prompt)

    logger.info("[Tool: extract_news_contents] Contents extracted successfully!")
    return response.content

@tool
def summarize_news_contents(extractd_content: str) -> str:
    """
    Take extracted contents, create a 3–4 sentence news summary in simple language that anyone can understand — no jargon
    Returns a concise, clear news summary
    """
    logger.info("[Tool: summarize_news_contents] Summarizing extracted news contents...")
    summarize_prompt = PromptTemplate(
        input_variables=["extractd_content"],
        template="""You are a senior news editor specialized in high-density factual synthesis..
        Contents: {extractd_content}
        Task: Analyze the provided extracted content and generate a structured summary. Do not add outside information or personal interpretation.
        Output Format:
        TL;DR (1 Sentence): A high-impact summary using the "Action-Impact-Context" formula.
        Key Points: 3-5 concise bullet points of verifiable facts.
        Important Data: List specific dates, figures, or names mentioned.
        Stakeholders: Identify the primary parties involved.
        Strict Constraints:
        Use clear action verbs (e.g., "Launched," "Rejected," "Secured").
        Remove all "fluff" like "The article discusses..." or "In this piece...".
        Maintain strict neutrality; do not include interpretation or bias."""
    )

    formatted_prompt = summarize_prompt.format(extractd_content=extractd_content)
    logger.info("[Tool: summarize_news_contents] Sending prompt to LLM for final sumarry...")

    response = llm.invoke(formatted_prompt)

    logger.info("[Tool: summarize_news_contents] Contents summarized successfully!")
    return response.content

tools = [extract_news_contents, summarize_news_contents]
logger.info(f"Tools registered: {[t.name for t in tools]}")

logger.info("Creating the agent...")
SYSTEM_PROMPT = """
Role: You are a News Automation Agent responsible for summarizing long news article contents in easy to read concise summary.
Core Objective: When a user gives you news article contents, use your defined tools extract core facts, and provide a structured summary by following these steps.:
First use extract_news_contents to extract verifiable facts (Who, What, Where, When, Why) while ignoring editorial bias. Then use summarize_news_contents
to summarize contents in easy-to-read summary.
Before outputting, think step-by-step to ensure the data is consistent across sources.
"""
agent_graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    debug=True
)

def run_content_summarizer(news_contents: str) -> str:
    """
    This function when invoked, it executes agent's ability to summarize long contents of news article in simple language summary.
    """

    logger.info("=" * 60)
    logger.info(f"PASTED NEWS ARTICLE CONTENTS: {news_contents}")
    logger.info("=" * 60)
    logger.info("Agent is now thinking... watch the tool-calling loop below!")
    logger.info("-" * 60)

    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=news_contents)]}
    )

    summarized_contents = result["messages"][-1].content

    logger.info("-" * 60)
    logger.info("Agent finished! Here's your summarized contents:")
    logger.info("=" * 60)

    return summarized_contents

# Main function
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  NEWS SUMMARIZER AGENT")
    print("  Powered by LangChain + OpenAI")
    print("=" * 60)
    print("\nPaste the contents of long news article, agent will extract the key facts and then writes a concise summary in simple, clear language.\n")
    print("Type 'quit' to exit.\n")




    while True:
        # news_contents = input("Please paste the contents of the news article: ")

        print("Paste your content and enter Ctrl+Z then enter to finish:")
        news_contents = sys.stdin.read()
        news_contents = news_contents.strip()

        if not news_contents:
            print("The contents of the news article required to move forward\n")
            continue

        if news_contents.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        try:
            summarized_contents = run_content_summarizer(news_contents)
            print("\n" + "=" * 60)
            print("YOUR SUMMARIZED CONTENTS:")
            print("=" * 60)
            print(summarized_contents)
            print("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Something went wrong: {e}")
            print(f"\nError: {e}")
            print("Please check your API key and try again.\n")


