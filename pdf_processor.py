import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    """
    Processes a PDF file by extracting text and splitting it into chunks.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of document chunks.
    """
    if not os.path.exists(file_path):
        return None

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {file_path}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks
