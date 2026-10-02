
# Run like this python -m configuration_health_check.testlangchain in root folder

from cofiguration.langchain_framework import get_agent

if __name__ == "__main__":
    my_agent = get_agent()
    inputs = {"messages": [{"role": "user", "content": "Who is AB in Bollywood?"}]}
    print(my_agent.invoke(inputs)["messages"][-1].content_blocks)
