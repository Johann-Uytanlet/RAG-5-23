#from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def pairwise_embedding_distance(query_embedding, document_embeddings):
    """
    Calculate pairwise cosine similarity between a query embedding and document embeddings.

    Args:
    - query_embedding (numpy.ndarray): The embedding of the query.
    - document_embeddings (list of numpy.ndarray): List of embeddings for documents.

    Returns:
    - numpy.ndarray: Array of cosine similarity scores between the query embedding and each document embedding.
    """
    # Calculate cosine similarity
    distances = cosine_similarity([query_embedding], document_embeddings)[0]

    return distances

def main():
    # Get embedding for a word.
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    #evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    vector2 = embedding_function.embed_query("iphone")
    #x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    x = pairwise_embedding_distance(vector, vector2)
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
