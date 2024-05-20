import os
import openai
import sys
import numpy as np
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global variables
docs = []  # List to store loaded documents
splits = []  # List to store split documents

def load_pdf(pdf_names):
    """
    Load PDF documents and split them into chunks.

    Args:
    pdf_names (list): List of PDF file paths to load.

    Returns:
    list: List of split document chunks.
    """
    global docs
    for pdf in pdf_names:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    return split_documents(docs)

def split_documents(docs):
    """
    Split documents into smaller chunks for processing.

    Args:
    docs (list): List of documents to split.

    Returns:
    list: List of split document chunks.
    """
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", "\. ", " ", ""]
    )
    return r_splitter.split_documents(docs)

def embed_and_store_splits(splits):
    """
    Embed and store document splits in Chroma.

    Args:
    splits (list): List of split document chunks.

    Returns:
    Chroma: Vector store with embedded documents.
    """
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
    )
    #vectordb.persist()
    return vectordb

def mmr_search(question, vectordb):
    """
    Perform MMR search and question answering on the vector store.

    Args:
    question (str): User's question.
    vectordb (Chroma): Vector store with embedded documents.

    Returns:
    str: Answer to the user's question.
    """
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever()
    )
    compressed_docs = compression_retriever.invoke(question)
    return question_answering(llm, compressed_docs, vectordb, question)

def question_answering(llm, compressed_docs, db, question):
    """
    Answer the user's question based on the compressed documents.

    Args:
    llm (OpenAI): OpenAI language model.
    compressed_docs (list): List of compressed documents.
    db (Chroma): Vector store with embedded documents.
    question (str): User's question.

    Returns:
    str: Answer to the user's question.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever()
    )
    result = qa_chain.invoke({"query": question})
    return result

def pretty_print_docs(docs):
    """
    Pretty print document contents.

    Args:
    docs (list): List of documents to print.
    """
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    print("\n")

def main():
    # Directory where example materials are located
    example_dir = "example_materials/"

    # Ask user to load PDF documents
    input_str = input("Enter the filenames of the PDF files (in the current directory or with paths), separated by commas: ")
    pdf_filenames = [filename.strip() for filename in input_str.split(',')]
    pdf_filenames = [filename if os.path.isabs(filename) else os.path.join(example_dir, filename) for filename in pdf_filenames]

    # Load and split the documents
    splits = load_pdf(pdf_filenames)
    print("Documents loaded and split into chunks.")

    # Embed and store the splits
    vectordb = embed_and_store_splits(splits)
    print("Documents embedded and stored.")

    while True:
        # Ask user a question about the documents
        question = input("Enter a question about the documents (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        # Perform MMR search and print results
        print("\n")
        results = mmr_search(question, vectordb)
        print(results["result"])
        print("\n")

if __name__ == "__main__":
    main()
