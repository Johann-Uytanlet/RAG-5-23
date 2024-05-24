from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"


def main():
    query_text = "How do Yangs help Langs"
    search_chroma_database(query_text)


def search_chroma_database(query_text):
    # Load the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Embed the query text (single item list)
    query_embedding = embeddings.embed_documents([query_text])[0]

    # Perform a similarity search with relevance scores
    results_with_scores = db.similarity_search_with_relevance_scores(query_text, k=5)

    # Print the results with relevance scores
    for i, (result, score) in enumerate(results_with_scores):
        print(f"Result {i + 1}:")
        print("Text:", result.page_content)
        print("Metadata:", result.metadata)
        print("Relevance Score:", score)
        print("----------")


if __name__ == "__main__":
    main()

