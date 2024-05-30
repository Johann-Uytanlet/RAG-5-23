from dataclasses import dataclass
import ollama
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import os
import shutil


# Query Translation


class RAG:
    def __init__(self, id, list_attr, description):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db_document = Chroma(persist_directory="chroma/document-level" , embedding_function=embeddings)

    # Simple Rag on topic 1 database    
    def QueryResponse(self, query_text, context=""):
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """
        all_documents = self.db_document.similarity_search_with_relevance_scores(query_text, k=5)
        context_text = context + "\n"
        context_text = context_text + "\n\n---\n\n".join([doc.page_content for doc, _score in all_documents])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        # print(prompt)

        # Use the ollama chat model
        response = ollama.chat(model='mistral', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])

        response_text = response['message']['content']

        sources = [doc.metadata.get("source", None) for doc, _score in all_documents]
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        QRAnswer = {}

        QRAnswer["query"] = query_text
        QRAnswer["prompt"] = prompt
        QRAnswer["context_text"] = context_text
        QRAnswer["context_source"] = sources
        QRAnswer["response"] = formatted_response
        QRAnswer["response_raw"] = response_text

        return QRAnswer

    def print_list(self):
        print(f"List: {self.list_attr}")

    def print_description(self):
        print(f"Description: {self.description}")


# Example usage:
obj = MyClass(1, [1, 2, 3], "This is a sample description.")
obj.print_id()  # Output: ID: 1
obj.print_list()  # Output: List: [1, 2, 3]
obj.print_description()  # Output: Description: This is a sample description.
