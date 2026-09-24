
# Run like this python -m langchain_features.testlangchain in root folder

from cofiguration.llm_provider import get_agent

if __name__ == "__main__":
    my_agent = get_agent()
    inputs = {"messages": [{"role": "user", "content": "Who is AB in Bollywood?"}]}
    print(my_agent.invoke(inputs)["messages"][-1].content_blocks)
