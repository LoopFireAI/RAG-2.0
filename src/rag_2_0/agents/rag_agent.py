"""
Simple RAG Agent for LangGraph Studio.
"""

from typing import TypedDict, List, Annotated, Literal
import operator
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import MessagesState
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from pydantic import BaseModel, Field

load_dotenv()

# State definition
class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    documents: List[str]
    context: str
    sources: List[str]  # Add sources to state
    grade: Literal["yes", "no"]
    is_social_media: bool  # Add flag for social media detection

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

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

SOCIAL_MEDIA_PROMPT = (
    "You are a social media content creator. Create a short, engaging post based on the following information. "
    "The post should be concise, use appropriate hashtags, and be engaging for social media. "
    "Keep it under 280 characters for Twitter/X compatibility.\n\n"
    "Information to use:\n{context}\n\n"
    "Create a social media post:"
)

def detect_social_media_request(state: RAGState) -> RAGState:
    """Detect if the query is requesting a social media post."""
    query = state["query"].lower()
    
    # Keywords that might indicate a social media post request
    social_media_keywords = [
        "tweet", "twitter", "post", "social media", "linkedin", "facebook",
        "instagram", "thread", "threads", "make a post", "create a post"
    ]
    
    is_social_media = any(keyword in query for keyword in social_media_keywords)
    return {"is_social_media": is_social_media}

def generate_social_media_post(state: RAGState) -> RAGState:
    """Generate a social media post based on the context."""
    query = state["query"]
    
    # First retrieve documents for the social media post
    results = vector_store.similarity_search(
        query, 
        k=int(os.getenv("TOP_K", 3))
    )
    
    documents = [doc.page_content for doc in results]
    context = "\n\n".join(documents)
    
    prompt = SOCIAL_MEDIA_PROMPT.format(context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [response]}

def grade_documents(state: RAGState) -> RAGState:
    """Grade the relevance of retrieved documents to the query."""
    query = state["query"]
    context = state["context"]
    
    prompt = GRADE_PROMPT.format(question=query, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Extract the grade from the response
    grade = "yes" if "yes" in response.content.lower() else "no"
    
    return {"grade": grade}

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
    
    # Extract sources from metadata
    sources = []
    for doc in results:
        if 'source' in doc.metadata:
            sources.append(doc.metadata['source'])

    # Debug print statements
    print(f"[DEBUG] Retrieved {len(results)} documents for query: '{query}'")
    for i, doc in enumerate(results):
        try:
            print(f"[DEBUG] Doc {i+1} content (first 100 chars): {doc.page_content[:100]}")
            if 'source' in doc.metadata:
                print(f"[DEBUG] Doc {i+1} source: {doc.metadata['source']}")
        except Exception as e:
            print(f"[DEBUG] Error reading content of doc {i+1}: {e}")
    
    return {
        "documents": documents,
        "context": context,
        "sources": sources
    }

def generate_response(state: RAGState) -> RAGState:
    """Generate response using retrieved context."""
    query = state["query"]
    context = state["context"]
    grade = state.get("grade", "yes")  # Default to "yes" if not present
    sources = state.get("sources", [])  # Get sources from state
    
    # Only use context if it was graded as relevant
    if grade == "yes":
        # Format sources as markdown links
        sources_text = "\n\nSources:\n" + "\n".join([f"- [View Document]({source})" for source in sources]) if sources else ""
        
        prompt = f"""Based on the following context, answer the question. Include relevant source links in your response.

Context:
{context}

Question: {query}

{sources_text}

Answer: Please provide your response and include the source links in markdown format like this: [Document Title](link)."""
    else:
        prompt = f"""I don't have enough relevant information to answer this question accurately.

Question: {query}

Answer: I apologize, but I don't have enough relevant information to provide a confident answer to your question."""
    
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
    workflow.add_node("detect_social_media", detect_social_media_request)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate_response)
    workflow.add_node("generate_social_media", generate_social_media_post)
    
    # Define workflow
    workflow.set_entry_point("extract_query")
    
    # Add conditional edge for social media detection
    workflow.add_edge("extract_query", "detect_social_media")
    workflow.add_conditional_edges(
        "detect_social_media",
        lambda x: "generate_social_media" if x["is_social_media"] else "retrieve",
        {
            "generate_social_media": "generate_social_media",
            "retrieve": "retrieve"
        }
    )
    
    # Add edges for regular RAG flow
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("generate_social_media", END)
    
    return workflow.compile()

# Create the graph for Studio
graph = create_rag_graph()