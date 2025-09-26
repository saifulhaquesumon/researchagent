import streamlit as st
import os
import tempfile
from agent import create_agent
from memory import LongTermMemory, ShortTermMemory
from pdf_processor import process_pdf
from retriever import create_retriever

# --- Page Configuration ---
st.set_page_config(
    page_title="Research Assistant AI",
    page_icon="🤖",
    layout="wide"
)

# --- Title and Description ---
st.title("🤖 Research Assistant AI Agent")
st.markdown("""
Welcome to your intelligent Research Assistant! 

Upload one or more PDF documents, and I'll help you extract information, answer questions, and summarize content. 
My memory works in two ways:
- **Short-Term Memory**: I'll remember our conversation during this session.
- **Long-Term Memory**: I'll retain key insights from the documents for future sessions.
""")

# --- Functions ---
@st.cache_resource
def get_long_term_memory():
    """Initializes and returns the long-term memory instance."""
    return LongTermMemory()

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if 'short_term_memory' not in st.session_state:
        st.session_state.short_term_memory = ShortTermMemory()
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your documents today?"}]
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []


# --- Main Application Logic ---
long_term_memory = get_long_term_memory()
initialize_session_state()

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF files here and click 'Process'",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                all_chunks = []
                newly_processed_files = []

                for uploaded_file in uploaded_files:
                    # Avoid reprocessing the same file in a session
                    if uploaded_file.name not in st.session_state.processed_files:
                        try:
                            # Save to a temporary file to get a path
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                temp_path = tmp_file.name
                            
                            # Process the PDF from the temporary path
                            chunks = process_pdf(temp_path)
                            if chunks:
                                all_chunks.extend(chunks)
                                newly_processed_files.append(uploaded_file.name)
                            
                            # Clean up the temporary file
                            os.remove(temp_path)

                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if all_chunks:
                    # Create retriever for the current session and update long-term memory
                    session_retriever = create_retriever(all_chunks)
                    long_term_memory.add_documents(all_chunks)
                    
                    # Create and store the agent in session state
                    st.session_state.agent = create_agent(
                        session_retriever,
                        st.session_state.short_term_memory.get_memory()
                    )
                    st.session_state.processed_files.extend(newly_processed_files)
                    st.success(f"Successfully processed: {', '.join(newly_processed_files)}")
                else:
                    st.warning("No new documents to process or failed to extract content.")
        else:
            st.warning("Please upload at least one PDF file.")

    st.header("Processed Files")
    if st.session_state.processed_files:
        for file_name in st.session_state.processed_files:
            st.info(f"📄 {file_name}")
    else:
        st.info("No files processed in this session yet.")


# --- Chat Interface ---
st.header("Chat with your Assistant")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.agent is None:
        st.warning("Please process at least one document before asking questions.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get conversation history for the agent
                memory = st.session_state.short_term_memory.get_memory()
                history_messages = memory.chat_memory.messages
                history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])

                # Get response from the agent
                response = st.session_state.agent.invoke({
                    "question": prompt, 
                    "history": history_str
                })
                
                st.write(response)
                
                # Update short-term memory and session state
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
