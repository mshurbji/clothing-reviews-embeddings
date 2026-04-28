import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def create_vector_db(texts):
    client = chromadb.PersistentClient()

    collection = client.create_collection(
        name="review_embeddings",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"]
        )
    )

    collection.add(
        documents=texts,
        ids=[str(i) for i in range(len(texts))]
    )

    return collection


def find_similar_reviews(input_text, collection, n=3):
    results = collection.query(
        query_texts=[input_text],
        n_results=n
    )
    return results["documents"][0]
