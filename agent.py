import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


def create_agent(retriever, short_term_memory):
    """
    Creates the AI agent with the specified retriever and memory.
    """

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("BASE_URL")
    openai_model = os.getenv("MODEL_NAME")


    llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
            model=openai_model,
            temperature=0.7
        )

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

    # --- THIS IS THE CORRECTED SECTION ---
    # We explicitly define how to process the input dictionary.
    # The retriever now correctly receives only the question string.
    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": (lambda x: x["question"]),
            "history": (lambda x: x["history"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser


# def create_agent(retriever, short_term_memory):
#     """
#     Creates the AI agent with the specified retriever and memory.
#     """

#     load_dotenv()
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     openai_base_url = os.getenv("BASE_URL")
#     openai_model = os.getenv("MODEL_NAME")


#     llm = ChatOpenAI(
#             api_key=openai_api_key,
#             base_url=openai_base_url,
#             model=openai_model,
#             temperature=0.7
#         )
#     #llm = ChatOpenAI(temperature=0.7)

#     template = """
#     You are a research assistant. Use the following pieces of context and conversation history to answer the question at the end.
#     Context: {context}
#     History: {history}
#     Question: {question}
#     Helpful Answer:
#     """
#     prompt = PromptTemplate(input_variables=["context", "history", "question"], template=template)

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return rag_chain


