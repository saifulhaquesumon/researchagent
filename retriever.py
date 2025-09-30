from langchain_community.vectorstores import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

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

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$1")
    print(chunks[0])
    # Convert dicts to Document objects if needed
    if isinstance(chunks[0], dict):
        chunks = [Document(page_content=d["content"]) for d in chunks]  # Adjust key if needed

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$2")
    print(chunks[0])

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$3")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(chunks
                                        , embedding_function
                                        , persist_directory="./chroma_db"
                                        )

    #vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore.as_retriever()
