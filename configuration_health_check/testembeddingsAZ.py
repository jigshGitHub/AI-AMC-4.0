# Run following command to test in root folder
# python -m configuration_health_check.testembeddingsAZ
import os
from cofiguration.azure_framework import get_openAIClient
from cofiguration import env_settings
if __name__ == "__main__":
    if os.system("cls" if os.name == "nt" else "clear") is not None:
        pass  # Clear the console for better readability
    client = get_openAIClient()
    embedding_response = client.embeddings.create(
        input=["Corporate search phrase query."],
        model=env_settings.TEXT_EMBEDDING_MODEL
    )
    print(embedding_response.data[0].embedding)
