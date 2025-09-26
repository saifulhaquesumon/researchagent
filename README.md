Research Assistant AI Agent with Memory
This project implements a sophisticated AI agent using LangGraph that can read and analyze PDF documents, answer user queries, and leverage both short-term and long-term memory for intelligent, context-aware responses.

Agent Architecture
The agent is built on a graph-based architecture using LangGraph, where each node in the graph represents a specific processing step. This modular design allows for a clear and manageable workflow, from processing user input to generating a final, informed answer.

The agent's workflow is as follows:

PDF Processing: The agent first processes the uploaded PDF, extracting text and chunking it for efficient analysis.

Information Retrieval: When a query is received, the agent retrieves relevant information from the processed document.

Memory Integration: The agent seamlessly integrates both short-term memory (for session-specific context) and long-term memory (for knowledge retained across sessions).

Intelligent Response Generation: By combining retrieved information with its memory, the agent generates accurate and contextually relevant answers to user queries.

Memory Design
Short-Term Memory
Short-term memory is designed to maintain the context of the current user session. It keeps track of the ongoing conversation, including user queries and the agent's responses. This allows the agent to understand follow-up questions and provide more natural and coherent interactions. This memory is volatile and is cleared at the end of each session.

Long-Term Memory
Long-term memory is implemented to retain valuable insights and knowledge across multiple sessions. This is achieved by storing key information, such as summaries and frequently accessed data, in a vector database (ChromaDB). This persistent memory allows the agent to build on its knowledge over time, becoming more effective with each interaction.

How to Run the Project
Install Dependencies:

pip install -r requirements.txt

Run the Main Application:

python main.py

You can then interact with the agent through the command-line interface, upload PDFs for analysis, and ask questions to see the agent in action.

Example Queries
Here are a few examples of how you can interact with the agent:

"What is the main topic of this document?"

"Summarize the key findings from the report."

"Based on our conversation, can you elaborate on your last point?"

"Have we discussed this topic in a previous session?"

These queries demonstrate the agent's ability to handle a range of tasks, from simple information retrieval to more complex, context-aware interactions that leverage its memory capabilities.