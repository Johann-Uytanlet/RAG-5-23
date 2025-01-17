from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"


def main():
    query_text = "Your query text here"
    search_chroma_database(query_text)


def search_chroma_database(query_text):
    # Load the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Embed the query text (single item list)
    query_embedding = embeddings.embed_documents([query_text])[0]

    # Perform a similarity search with precomputed embeddings
    results = db.similarity_search_by_vector(query_embedding, k=5)  # Get top 5 most similar documents
    #results = db.similarity_search(query_text, k=5)

    # Print the results
    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        print("Text:", result.page_content)
        print("Metadata:", result.metadata)
        print("----------")


if __name__ == "__main__":
    main()
