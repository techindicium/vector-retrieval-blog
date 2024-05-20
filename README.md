# LangChain Vector Store Retriever Tutorial

## Introduction

This repository demonstrates how to use a Vector Store retriever in a conversational chain with LangChain, comparing two popular vector stores: LanceDB and Chroma. These tools help manage and retrieve data efficiently, making them essential for AI applications.

## Setup Instructions

### 1. Create a Virtual Environment

First, create a virtual environment to manage your project dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
```

### 2. Install Dependencies

Install the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in your project directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

### 4. Run the Script

Execute the `langchain_agent.py` script to load documents, embed them into a vector store, and perform retrieval operations.

```bash
python langchain_agent.py
```

## Script Overview

### Environment Setup

The script begins by loading environment variables from a `.env` file and setting up the OpenAI API key.

```python
import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')
```

### Global Variables

Two global lists are used to store documents and their splits.

```python
docs = []  # List to store loaded documents
splits = []  # List to store split documents
```

### Functions

#### `load_pdf(pdf_names)`

This function is responsible for loading PDF documents from the given file paths and splitting them into smaller chunks. It uses the `PyPDFLoader` to read each PDF file and store its contents in the global `docs` list. After loading all documents, it calls the `split_documents` function to process them into manageable chunks.

```python
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
```

#### `split_documents(docs)`

This function splits the loaded documents into smaller, more manageable chunks. It uses the `RecursiveCharacterTextSplitter` to divide each document based on specified chunk size and overlap. The chunk overlap ensures that important context is preserved across chunks, improving the relevance of retrieval operations.

```python
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
```

#### `embed_and_store_splits(splits)`

This function embeds the split document chunks into high-dimensional vectors using OpenAI embeddings. It then stores these vectors in a Chroma vector store. This process allows for efficient similarity searches and retrieval operations based on the semantic content of the documents.

```python
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
    return vectordb
```

#### `mmr_search(question, vectordb)`

This function performs a Maximal Marginal Relevance (MMR) search on the vector store to retrieve and answer questions based on the embedded documents. It uses a language model to compress the retrieved documents and enhance the relevance of the results. The compressed documents are then used to generate a precise answer to the user's query.

```python
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
```

#### `question_answering(llm, compressed_docs, db, question)`

This function generates an answer to the user's question by leveraging the compressed documents retrieved from the vector store. It constructs a retrieval-based question-answering chain using the language model and retrieves the most relevant information from the vector store to provide an accurate response.

```python
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
```

#### `pretty_print_docs(docs)`

This function prints the contents of the documents in a readable format. It is useful for debugging and verifying that the documents have been correctly loaded and split.

```python
def pretty_print_docs(docs):
    """
    Pretty print document contents.

    Args:
    docs (list): List of documents to print.
    """
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    print("\n")
```

### Main Function

The `main()` function orchestrates the loading, splitting, embedding, and querying of documents. It prompts the user to enter filenames of PDFs, loads and splits the documents, embeds them into Chroma, and allows the user to ask questions about the documents to retrieve relevant information.

```python
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
```

### Conclusion

This repository provides a comprehensive tutorial on using Vector Store retrievers with LangChain, demonstrating the capabilities of LanceDB and Chroma. Each tool has its strengths and is suited to different types of projects, making this tutorial a valuable resource for understanding and implementing vector retrieval in AI applications.

For further reading and resources, check out the LangChain Documentation and the DeepLearning.AI LangChain Course.
