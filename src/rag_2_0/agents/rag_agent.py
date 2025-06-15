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
    detected_leader: str  # Add detected leader name
    tone_profile: str  # Add tone profile content
    # Feedback fields
    response_id: str  # Unique response identifier for feedback correlation
    feedback_collected: bool  # Track if feedback was collected
    retrieved_docs_metadata: List[dict]  # Store document metadata for feedback
    # Conversation state
    waiting_for_leader: bool  # Track if we're waiting for leader specification
    original_query: str  # Store the original query when waiting for leader

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

def load_tone_profile(leader_name: str) -> str:
    """Load tone profile from markdown file."""
    from pathlib import Path
    
    # Get the directory where this file is located
    current_dir = Path(__file__).parent.parent
    tones_dir = current_dir / "tones"
    
    # Try to load the specific leader's tone file
    tone_file = tones_dir / f"{leader_name.lower()}.md"
    
    if tone_file.exists():
        return tone_file.read_text(encoding='utf-8')
    else:
        # Fallback to default tone
        default_file = tones_dir / "default.md"
        if default_file.exists():
            return default_file.read_text(encoding='utf-8')
        else:
            return "Use a professional and helpful tone."

def detect_tone_and_leader(state: RAGState) -> RAGState:
    """Detect leader mention and load appropriate tone profile."""
    query = state["query"].lower()
    
    # Available leaders (you can add more here)
    leaders = ["janelle", "doreen"]  # Add the actual second leader name when you know it
    
    detected_leader = "default"
    
    # Check for leader mentions in various formats
    for leader in leaders:
        leader_patterns = [
            f"as {leader}",
            f"like {leader}",
            f"{leader} would",
            f"in {leader}'s voice",
            f"{leader}'s style",
            f"how would {leader}",
            f"@{leader}",
            leader
        ]
        
        if any(pattern in query for pattern in leader_patterns):
            detected_leader = leader
            break
    
    # Load the tone profile
    tone_profile = load_tone_profile(detected_leader)
    
    return {
        "detected_leader": detected_leader,
        "tone_profile": tone_profile
    }

def generate_social_media_post(state: RAGState) -> RAGState:
    """Generate a social media post based on the context and tone."""
    query = state["query"]
    detected_leader = state.get("detected_leader", "default")
    tone_profile = state.get("tone_profile", "Use a professional and helpful tone.")
    
    # First retrieve documents for the social media post
    results = vector_store.similarity_search(
        query, 
        k=int(os.getenv("TOP_K", 3))
    )
    
    documents = [doc.page_content for doc in results]
    context = "\n\n".join(documents)
    
    prompt = f"""You are tasked with creating a social media post reflecting {detected_leader}'s unique voice and style.

Tone & Communication Style:
{tone_profile}

Guidelines:
- Craft a concise, engaging post suitable for social media platforms like LinkedIn and X.
- Use a natural tone that aligns with {detected_leader}'s communication style.
- Incorporate examples from the {detected_leader}'s work.
- Focus on making the content clear, compelling, and in line with the provided context.

Context:
{context}

Please generate a social media post in {detected_leader}'s voice that meets these criteria:"""
    
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
    """Retrieve relevant documents with feedback-enhanced scoring."""
    query = state["query"]
    
    # Get feedback storage for document scoring enhancement
    try:
        from rag_2_0.feedback.feedback_storage import FeedbackStorage
        feedback_storage = FeedbackStorage()
    except ImportError:
        feedback_storage = None
    
    # Use LangChain Chroma similarity search
    results = vector_store.similarity_search(
        query, 
        k=int(os.getenv("TOP_K", 3))
    )
    
    # Apply feedback-based reranking if available
    if feedback_storage and results:
        doc_ids = [doc.metadata.get('id', doc.metadata.get('source', '')) for doc in results]
        feedback_scores = feedback_storage.get_document_feedback_scores(doc_ids)
        
        # Rerank based on feedback (boost good docs, demote bad ones)
        for i, doc in enumerate(results):
            doc_id = doc.metadata.get('id', doc.metadata.get('source', ''))
            if doc_id in feedback_scores:
                feedback_score = feedback_scores[doc_id]
                # Boost/demote based on feedback (3.0 is neutral)
                boost_factor = (feedback_score - 3.0) * 0.1
                # This is a conceptual boost - in practice, you'd adjust the similarity scores
                doc.metadata['feedback_boost'] = boost_factor
    
    documents = [doc.page_content for doc in results]
    context = "\n\n".join(documents)
    
    # Extract sources and metadata for feedback
    sources = []
    retrieved_docs_metadata = []
    for doc in results:
        if 'source' in doc.metadata:
            sources.append(doc.metadata['source'])
        
        # Store full metadata for feedback correlation
        retrieved_docs_metadata.append({
            'id': doc.metadata.get('id', doc.metadata.get('source', '')),
            'title': doc.metadata.get('title', doc.metadata.get('source', 'Unknown')),
            'source': doc.metadata.get('source', ''),
            'metadata': doc.metadata,
            'content_preview': doc.page_content[:200]
        })

    # Debug print statements
    print(f"[DEBUG] Retrieved {len(results)} documents for query: '{query}'")
    for i, doc in enumerate(results):
        try:
            print(f"[DEBUG] Doc {i+1} content (first 100 chars): {doc.page_content[:100]}")
            if 'source' in doc.metadata:
                print(f"[DEBUG] Doc {i+1} source: {doc.metadata['source']}")
            if 'feedback_boost' in doc.metadata:
                print(f"[DEBUG] Doc {i+1} feedback boost: {doc.metadata['feedback_boost']}")
        except Exception as e:
            print(f"[DEBUG] Error reading content of doc {i+1}: {e}")
    
    return {
        "documents": documents,
        "context": context,
        "sources": sources,
        "retrieved_docs_metadata": retrieved_docs_metadata
    }

def generate_response(state: RAGState) -> RAGState:
    """Generate response using retrieved context and tone profile."""
    import uuid
    import time
    
    start_time = time.time()
    query = state["query"]
    context = state["context"]
    grade = state.get("grade", "yes")  # Default to "yes" if not present
    sources = state.get("sources", [])  # Get sources from state
    detected_leader = state.get("detected_leader", "default")
    tone_profile = state.get("tone_profile", "Use a professional and helpful tone.")
    
    # Generate unique response ID for feedback correlation
    response_id = str(uuid.uuid4())
    
    # Only use context if it was graded as relevant
    if grade == "yes":
        # Format sources as markdown links
        sources_text = "\n\nSources:\n" + "\n".join([f"- [View Document]({source})" for source in sources]) if sources else ""
        
        prompt = f"""You are responding as {detected_leader}. Use the following tone and communication style:

{tone_profile}

Based on the following context, answer the question in the specified tone and style. Include relevant source links in your response.

Context:
{context}

Question: {query}

{sources_text}

Answer: Please provide your response in {detected_leader}'s communication style and include the source links in markdown format like this: [Document Title](link)."""
    else:
        prompt = f"""You are responding as {detected_leader}. Use the following tone and communication style:

{tone_profile}

I don't have enough relevant information to answer this question accurately.

Question: {query}

Answer: I apologize, but I don't have enough relevant information to provide a confident answer to your question. Please respond in {detected_leader}'s communication style."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Log token usage if available
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        token_usage = response.response_metadata['token_usage']
        print(f"Token usage - Input: {token_usage.get('prompt_tokens', 0)}, Output: {token_usage.get('completion_tokens', 0)}, Total: {token_usage.get('total_tokens', 0)}")
    
    # Store response time for feedback tracking
    state_update = {
        "messages": [response],
        "response_id": response_id,
        "feedback_collected": False
    }
    
    # Add response time to retrieved docs metadata for feedback
    if "retrieved_docs_metadata" in state:
        for doc_meta in state["retrieved_docs_metadata"]:
            doc_meta["response_time_ms"] = response_time_ms
    
    return state_update

def register_response_for_feedback(state: RAGState) -> RAGState:
    """Register response with feedback collector for potential feedback collection."""
    try:
        from rag_2_0.feedback.feedback_collector import FeedbackCollector
        from rag_2_0.feedback.feedback_storage import FeedbackStorage
        
        print("[DEBUG] Attempting to register response for feedback...")
        
        # Initialize feedback system
        storage = FeedbackStorage()
        collector = FeedbackCollector(storage)
        
        # Get response content
        response_content = ""
        if state.get("messages"):
            last_message = state["messages"][-1]
            response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Register the response for feedback
        response_id = state.get("response_id", "")
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs_metadata", [])
        
        print(f"[DEBUG] Response ID: {response_id}")
        print(f"[DEBUG] Query: {query[:50]}...")
        print(f"[DEBUG] Response content length: {len(response_content)}")
        print(f"[DEBUG] Retrieved docs: {len(retrieved_docs)}")
        
        if response_id and response_content and query:
            registered_id = collector.register_response(
                query=query,
                response=response_content,
                retrieved_docs=retrieved_docs,
                persona=state.get("detected_leader", "default"),
                response_time_ms=retrieved_docs[0].get("response_time_ms", 0) if retrieved_docs else 0,
                response_id=response_id  # Use the existing response_id
            )
            print(f"[DEBUG] Successfully registered response with ID: {registered_id}")
        else:
            print(f"[DEBUG] Failed to register: missing data - response_id={bool(response_id)}, content={bool(response_content)}, query={bool(query)}")
        
    except ImportError:
        print("[DEBUG] Feedback system not available (ImportError)")
    except Exception as e:
        print(f"[DEBUG] Error registering response for feedback: {e}")
        import traceback
        traceback.print_exc()
    
    return state

def elicit_leader(state: RAGState) -> RAGState:
    """Elicit leader specification if not provided in the query."""
    query = state["query"]
    messages = state.get("messages", [])
    
    # First, try to detect if a leader is already specified
    detection_prompt = f"""Analyze the following query and determine if a specific leader's voice is requested.
    Available leaders: Janelle (strategic business perspective) and Doreen (relational, equity-focused approach).
    
    Query: {query}
    
    Respond with either:
    1. The leader's name if specified (janelle/doreen)
    2. "ask" if no leader is specified
    
    Just respond with the leader name or "ask"."""
    
    detection_response = llm.invoke([HumanMessage(content=detection_prompt)])
    response_content = detection_response.content.strip().lower()
    
    # If no leader is specified, use a simple follow-up question
    if response_content == "ask":
        follow_up = "Which leader's voice would you prefer - Janelle or Doreen?"
        return {
            "messages": [AIMessage(content=follow_up)],
            "waiting_for_leader": True,
            "original_query": query
        }
    
    # If a leader is detected, proceed with the normal flow
    return {
        "detected_leader": response_content,
        "waiting_for_leader": False
    }

def create_rag_graph():
    """Create the RAG workflow graph."""
    
    # Create workflow
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("extract_query", extract_query)
    workflow.add_node("detect_social_media", detect_social_media_request)
    workflow.add_node("elicit_leader", elicit_leader)
    workflow.add_node("detect_tone", detect_tone_and_leader)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate_response)
    workflow.add_node("generate_social_media", generate_social_media_post)
    workflow.add_node("register_feedback", register_response_for_feedback)
    
    # Define workflow
    workflow.set_entry_point("extract_query")
    
    # After extract_query, go to detect_social_media
    workflow.add_edge("extract_query", "detect_social_media")
    
    # After detect_social_media, branch:
    workflow.add_conditional_edges(
        "detect_social_media",
        lambda x: "elicit_leader" if x["is_social_media"] else "retrieve",
        {
            "elicit_leader": "elicit_leader",
            "retrieve": "retrieve"
        }
    )
    
    # After elicit_leader, branch:
    workflow.add_conditional_edges(
        "elicit_leader",
        lambda x: "detect_tone" if not x.get("waiting_for_leader", False) else END,
        {
            "detect_tone": "detect_tone",
            END: END
        }
    )
    
    # After detect_tone, go to generate_social_media
    workflow.add_edge("detect_tone", "generate_social_media")
    # After generate_social_media, go to register_feedback
    workflow.add_edge("generate_social_media", "register_feedback")
    
    # Non-social media path
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", "register_feedback")
    
    # End
    workflow.add_edge("register_feedback", END)
    
    return workflow.compile()

# Create the graph for Studio
graph = create_rag_graph()