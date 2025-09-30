from langchain_community.vectorstores import Chroma
# Updated import for HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

class LongTermMemory:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        # This is the key change: Use the HuggingFaceEmbeddings wrapper
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )

    def add_documents(self, chunks):
        """
        Adds document chunks to the long-term memory.
        """
        if chunks:
            self.vectorstore.add_documents(chunks)
            # Persisting is handled by Chroma automatically on add, but explicit is fine
            self.vectorstore.persist()

    def get_retriever(self):
        """
        Returns a retriever for the long-term memory.
        """
        return self.vectorstore.as_retriever()

class ShortTermMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)

    def get_memory(self):
        """
        Returns the short-term conversation memory.
        """
        return self.memory