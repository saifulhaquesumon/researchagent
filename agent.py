from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_agent(retriever, short_term_memory):
    """
    Creates the AI agent with the specified retriever and memory.
    """
    llm = ChatOpenAI(temperature=0.7)

    template = """
    You are a research assistant. Use the following pieces of context and conversation history to answer the question at the end.
    Context: {context}
    History: {history}
    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "history", "question"], template=template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
