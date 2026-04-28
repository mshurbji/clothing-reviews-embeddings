import pandas as pd
import numpy as np

from src.embeddings import create_embeddings
from src.analysis import apply_tsne, plot_tsne
from src.similarity import create_vector_db, find_similar_reviews


def main():
    # Load data
    reviews = pd.read_csv("data/womens_clothing_e-commerce_reviews.csv")
    review_texts = reviews["Review Text"].dropna().tolist()

    # Create embeddings
    embeddings = create_embeddings(review_texts)

    # Save embeddings
    np.save("output/embeddings.npy", embeddings)

    # Apply t-SNE
    embeddings_2d = apply_tsne(embeddings)
    np.save("output/embeddings_2d.npy", embeddings_2d)

    # Plot
    plot_tsne(embeddings_2d)

    # Create vector DB
    collection = create_vector_db(review_texts)

    # Example similarity search
    example = "Absolutely wonderful - silky and comfortable"
    similar = find_similar_reviews(example, collection, 3)

    print("\nSimilar reviews:\n")
    for s in similar:
        print("-", s)


if __name__ == "__main__":
    main()
