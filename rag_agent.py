"""
Simple RAG Agent for LangGraph Studio.
"""

from typing import TypedDict, List, Annotated
import operator
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# State definition
class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    documents: List[str]
    context: str

# Initialize components
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    temperature=float(os.getenv("TEMPERATURE", 0.1)),
    max_tokens=int(os.getenv("MAX_TOKENS", 1000)),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize vector store
vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

def extract_query(state: RAGState) -> RAGState:
    """Extract query from messages."""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        # Handle both dict and LangChain message formats
        if hasattr(last_message, 'content'):
            query = last_message.content
        elif isinstance(last_message, dict):
            query = last_message.get('content', '')
        else:
            query = str(last_message)
    else:
        query = "What is machine learning?"
    
    return {"query": query}

def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents."""
    query = state["query"]
    
    # Use LangChain Chroma similarity search
    results = vector_store.similarity_search(
        query, 
        k=int(os.getenv("TOP_K", 3))
    )
    
    documents = [doc.page_content for doc in results]
    context = "\n\n".join(documents)
    
    return {
        "documents": documents,
        "context": context
    }

def generate_response(state: RAGState) -> RAGState:
    """Generate response using retrieved context."""
    query = state["query"]
    context = state["context"]
    
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Log token usage if available
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        token_usage = response.response_metadata['token_usage']
        print(f"Token usage - Input: {token_usage.get('prompt_tokens', 0)}, Output: {token_usage.get('completion_tokens', 0)}, Total: {token_usage.get('total_tokens', 0)}")
    
    return {"messages": [response]}

def create_rag_graph():
    """Create the RAG workflow graph."""
    
    # Create workflow
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("extract_query", extract_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_response)
    
    # Define workflow
    workflow.set_entry_point("extract_query")
    workflow.add_edge("extract_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Create the graph for Studio
graph = create_rag_graph()