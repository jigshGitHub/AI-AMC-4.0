# Run following command to test in root folder
# python -m configuration_health_check.testembeddingsAZ
from cofiguration.azure_framework import get_azOpenAIClient, get_openAIClient
if __name__ == "__main__":
    # client = get_azOpenAIClient()
    # embedding_response = client.embeddings.create(
    #     input=["Corporate search phrase query."],
    #     model="text-embedding-3-small"
    # )

    client = get_openAIClient()
    embedding_response = client.embeddings.create(
        input=["Corporate search phrase query."],
        model="text-embedding-3-small"
    )
    print(embedding_response.data[0].embedding)
