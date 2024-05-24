from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/gen-ai-topic1-corpus"


def main():
    # documents are split by documents
    documents = load_documents("data/gen-ai-topic1-corpus")
    #save_to_chroma(documents, "chroma/document-level")

    # documents are split by line breaks
    chunks = split_documents_into_chunks(documents)

    # Print the chunks
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}:")
        print("Text:", chunk.page_content)
        print("Metadata:", chunk.metadata)
        print("----------")

    #generate_data_store()
    #documents = load_documents()
    #for document in documents:
    #    print("content:")
    #    print(document.page_content)
    #    print("meta data")
    #    print(document.metadata)
    #generate_data_store()


def generate_data_store(data_path, chroma_path):
    documents = load_documents(data_path)
    # Documents are relatively small so no need to split?
    #chunks = split_text(documents)
    save_to_chroma(documents, chroma_path)


def load_documents(data_path):
    loader = DirectoryLoader(data_path, glob="*.txt")
    documents = loader.load()
    return documents

def split_documents_into_chunks(documents):
    chunks = []
    for document in documents:
        content = document.page_content.strip()  # Remove leading and trailing whitespace
        document_lines = content.split("\n")  # Split the content into lines
        start_index = document.metadata.get("start_index", 0)  # Get the start index from metadata, default to 0 if not present
        for index, line in enumerate(document_lines, start=start_index):
            if line.strip():  # Skip empty lines
                # Create a new document for each line with the original metadata and start index
                chunk_metadata = document.metadata.copy()  # Copy original metadata
                chunk_metadata["start_index"] = index  # Update start index
                chunk = Document(page_content=line, metadata=chunk_metadata)
                chunks.append(chunk)
    return chunks

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

#  change this
def save_to_chroma(chunks: list[Document], chroma_path):
    # Clear out the database first.
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Create a new DB from the documents.
    #db = Chroma.from_documents(
    #    chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    #)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")


if __name__ == "__main__":
    main()
