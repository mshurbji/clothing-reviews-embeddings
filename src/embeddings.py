import openai

EMBEDDING_MODEL = "text-embedding-3-small"

def create_embeddings(texts):
    client = openai.OpenAI()
    responses = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    ).model_dump()

    embeddings = [r["embedding"] for r in responses["data"]]
    return embeddings
