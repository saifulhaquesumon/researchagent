import os
from agent import create_agent
from memory import LongTermMemory, ShortTermMemory
from pdf_processor import process_pdf
from retriever import create_retriever

def main():
    """
    Main function to run the research assistant agent.
    """
    # Initialize memories
    long_term_memory = LongTermMemory()
    short_term_memory = ShortTermMemory()

    while True:
        print("\n--- Research Assistant ---")
        pdf_path = input("Enter the path to a PDF file (or 'exit'): ").strip()

        if pdf_path.lower() == 'exit':
            break

        if not os.path.exists(pdf_path):
            print("File not found. Please try again.")
            continue

        # Process the PDF
        chunks = process_pdf(pdf_path)
        if not chunks:
            print("Could not process the PDF.")
            continue

        # Create a session-specific retriever and update long-term memory
        session_retriever = create_retriever(chunks)
        long_term_memory.add_documents(chunks)

        # Create the agent
        agent = create_agent(session_retriever, short_term_memory.get_memory())

        # Conversational loop
        while True:
            query = input("\nAsk a question about the document (or 'back' to change PDF): ").strip()
            if query.lower() == 'back':
                break

            # Get conversation history
            history = short_term_memory.get_memory().chat_memory.messages
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])

            # Get response from the agent
            response = agent.invoke({"question": query, "history": history_str})
            print("\nAgent:", response)

            # Update short-term memory
            short_term_memory.get_memory().chat_memory.add_user_message(query)
            short_term_memory.get_memory().chat_memory.add_ai_message(response)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # To run this, you need to set your OpenAI API key
    # For example: export OPENAI_API_KEY="your_key_here"
    if os.getenv("OPENAI_API_KEY") is None:
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        main()
