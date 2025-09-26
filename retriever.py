from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_retriever(chunks):
    """
    Creates a retriever from a list of document chunks.

    Args:
        chunks (list): A list of document chunks.

    Returns:
        Chroma.as_retriever: A retriever object.
    """
    if not chunks:
        return None

    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore.as_retriever()
