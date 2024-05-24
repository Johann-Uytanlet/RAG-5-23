from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_documents(data_path):
    loader = DirectoryLoader(data_path, glob="*.txt")
    documents = loader.load()
    return documents

def split_documents_into_chunks(documents):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def main():
    documents = load_documents("data/gen-ai-topic1-corpus")
    chunks = split_documents_into_chunks(documents)
    # Print the chunks
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}:")
        print("Text:", chunk.page_content)
        print("Metadata:", chunk.metadata)
        print("----------")

if __name__ == "__main__":
    main()
