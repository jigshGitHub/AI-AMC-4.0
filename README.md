# 📰 News Summarizer Agent

An intelligent AI agent built with **LangChain** and **OpenAI** that extracts key factual data from long news articles and synthesizes them into concise, jargon-free summaries.

## 🚀 Features

- **Two-Stage Processing**: Uses a specialized tool-calling agent to first extract raw facts and then generate a high-density summary.
- **Fact-Focused Extraction**: Filters out editorial bias to focus on the Who, What, Where, When, and Why.
- **Multi-line Support**: Custom input handling allows you to paste long, multi-paragraph articles directly into the terminal.
- **Logging & Debugging**: Integrated logging to track agent reasoning and tool execution in real-time.

## 🛠️ Architecture

The system uses a **LangChain Agent** equipped with two custom tools:
1.  `extract_news_contents`: Distills raw data, stakeholders, and statistics from the source text.
2.  `summarize_news_contents`: Converts extracted data into a "Action-Impact-Context" summary.

## 📋 Prerequisites

- Python 3.9+
- OpenAI API Key

## ⚙️ Installation & Setup

1️⃣ Clone the Repository    
    git clone https://github.com/jigshGitHub/AI-AMC-4.0
    
2️⃣ Create a Virtual Environment
    VS Code / PowerShell:
        python -m venv labenv

3️⃣ Activate the Virtual Environment
    powershell
        ./labenv/bin/Activate.ps1
    vscode
        .\labenv\Scripts\activate
    💡  Note: If blocked, run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process first

4️⃣ Install Dependencies
    pip install -r requirements.txt

5️⃣ Configure Environment Variables
    Copy the example environment file
        cp .env.example .env
    Edit .env and enter your credentials
        OPENAI_API_KEY=your_sk_key_here
        GPT_MODEL=gpt-4o

6️⃣ Change Directory
    cd news_summarizer

7️⃣ Execute the Agent
    python run.py

🖥️ How to Use
Copy the full text of a news article.
Paste it into the terminal prompt.
Press ENTER THRICE to signal you are finished pasting.
Type quit or exit to close the agent