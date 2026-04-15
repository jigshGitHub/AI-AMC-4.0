import logging
import sys
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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

def run_content_summarizer(news_contents: str) -> str:
    """
    This function when invoked, it executes agent's ability to summarize long contents pf news article in simple language summary.
    """

    logger.info("=" * 60)
    logger.info(f"PASTED NEWS ARTICLE CONTENTS: {news_contents}")
    logger.info("=" * 60)
    logger.info("Agent is now thinking... watch the tool-calling loop below!")
    logger.info("-" * 60)

    # result = agent_graph.invoke(
    #     {"messages": [HumanMessage(content=news_contents)]}
    # )

    # summarized_contents = result["messages"][-1].content

    logger.info("-" * 60)
    logger.info("Agent finished! Here's your summarized contents:")
    logger.info("=" * 60)

    return news_contents # summarized_contents

# MMain function
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NEWS SUMMARIZER AGENT")
    print("  Powered by LangChain + OpenAI")
    print("=" * 60)
    print("\nPaste the contents of long news article, agent will extract the key facts and then writes a concise summary in simple, clear language.\n")
    print("Type 'quit' to exit.\n")

    while True:
        news_contents = input("Please paste the contents of the news article: ").strip()

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