from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

class LongTermMemory:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.embedding_function = OpenAIEmbeddings()
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
