"""
Simple RAG Agent for LangGraph Studio.
"""

from typing import TypedDict, List, Annotated, Literal
import operator
import os
import asyncio
import hashlib

from langchain_core.messages import BaseMessage, HumanMessage
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
    waiting_for_feedback: bool  # Track if we're waiting for user feedback

# Initialize components
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    temperature=float(os.getenv("TEMPERATURE", 0.1)),
    max_tokens=int(os.getenv("MAX_TOKENS", 1000)),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize vector store with environment-aware persistence
persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory=persist_dir
)

GRADE_PROMPT = """You are an expert content evaluator assessing document relevance with precision.

DOCUMENT CONTENT:
{context}

USER QUERY: 
{question}

EVALUATION CRITERIA:
1. DIRECT RELEVANCE: Does the document directly address the query topic?
2. CONCEPTUAL ALIGNMENT: Are the core concepts/themes aligned?
3. ACTIONABLE CONTENT: Does it provide information that can answer the query?
4. SPECIFICITY MATCH: Does the detail level match what's being asked?

DECISION FRAMEWORK:
- "yes" = Document contains substantial relevant content that directly helps answer the query
- "no" = Document lacks relevant content or only tangentially relates to the query

Think step-by-step:
1. Identify key concepts in the query
2. Scan document for matching concepts/examples
3. Assess if document provides actionable information for the query

RESPONSE: Return only "yes" or "no" """

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


def generate_social_media_post(state: RAGState) -> RAGState:
    """Generate a social media post based on the retrieved context and tone."""
    query = state["query"]
    context = state.get("context", "")
    grade = state.get("grade", "yes")
    detected_leader = state.get("detected_leader", "default")
    tone_profile = state.get("tone_profile", "Use a professional and helpful tone.")
    
    # Only use context if it was graded as relevant
    if grade == "no":
        context = "No relevant information found in the knowledge base."
    
    import uuid
    import time
    
    start_time = time.time()
    
    # Generate unique response ID for feedback correlation
    response_id = str(uuid.uuid4())
    
    # Determine post length and platform optimization based on query
    is_long_form = "linkedin" in query.lower() or "detailed" in query.lower() or "examples" in query.lower()
    char_limit = "400-500 characters" if is_long_form else "200-280 characters"
    platform_focus = "LinkedIn" if is_long_form else "X/Twitter"
    
    prompt = f"""You are {detected_leader}, crafting a thoughtful social media post that demonstrates your expertise while being genuinely engaging. You need to balance intelligence with relatability.

YOUR AUTHENTIC VOICE ({detected_leader.upper()}):
{tone_profile}

KNOWLEDGE BASE INSIGHTS:
{context}

REQUEST:
{query}

WRITING GUIDELINES:
- Strike the perfect balance: intelligent but not academic, relatable but not dumbed-down
- Use {detected_leader}'s signature thinking patterns and language style
- Ground every insight in SPECIFIC examples from the knowledge base
- Make complex ideas accessible without losing sophistication
- Sound like a thought leader having an authentic moment of insight

POST STRUCTURE:
1. COMPELLING OPENER: Start with a bold insight, surprising data point, or strategic observation that demonstrates your expertise

2. KNOWLEDGE DELIVERY: Present 3 SPECIFIC insights from the research as valuable knowledge you're sharing - focus on teaching and revealing patterns, not asking questions

3. STRATEGIC SYNTHESIS: Connect these insights to show the bigger systemic pattern or business implication - this is where {detected_leader}'s strategic thinking shines

4. SINGLE ENGAGEMENT HOOK: End with ONE well-crafted question or call-to-action that invites meaningful response

CONTENT APPROACH:
- Lead with knowledge and insights, not questions
- Each insight should feel like valuable information you're generously sharing
- Use declarative statements that show expertise: "Here's what the data reveals..." "The pattern I'm seeing..." "What's fascinating is..."
- Save questions for the very end - just one powerful engagement hook

VOICE CALIBRATION FOR {detected_leader.upper()}:
- Use strategic language that shows business acumen
- Include growth/scale thinking where natural
- Reference systems, patterns, and implications
- Sound like someone who sees the bigger picture
- Maintain authority while being approachable

LANGUAGE PATTERNS TO USE:
- "Here's what the research reveals..."
- "I've been analyzing patterns in..."
- "The data shows something fascinating..."
- "What strikes me about this trend..."
- "The strategic implication here is..."

FORMATTING RULES:
- NO asterisks, bold text, or special characters (**text** is forbidden)
- NO numbered lists (1. 2. 3.) or bullet points  
- NO headers, subheadings, or section labels
- Write in clean, flowing paragraphs like natural social media content
- Use simple line breaks for readability, nothing else

FORBIDDEN:
- Multiple questions throughout the post
- ANY formatting symbols: **, ##, •, 1., 2., etc.
- Section headers like "Identity Influence:" or "Key Insight:"
- Numbered or bulleted lists of any kind
- Academic or corporate-style organization
- More than ONE question (save it for the very end)

WRITING STYLE:
- Write like you're having an intelligent conversation
- Flow naturally from one insight to the next
- Use transitions like "What's fascinating is..." "Here's what caught my attention..." "The pattern I'm seeing..."
- Keep it conversational but sophisticated

GOAL: Create clean, readable social media content that flows naturally and teaches valuable insights, ending with ONE compelling engagement question."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Update retrieved docs metadata with response time for feedback
    if "retrieved_docs_metadata" in state:
        for doc_meta in state["retrieved_docs_metadata"]:
            doc_meta["response_time_ms"] = response_time_ms
    
    return {
        "messages": [response],
        "response_id": response_id,
        "feedback_collected": False
    }

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
    """Extract query from messages and reset state for new conversations."""
    messages = state.get("messages", [])
    
    # Check if this is a new conversation (only one message, which should be the user's query)
    is_new_conversation = len(messages) == 1
    
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
    
    # For new conversations, reset all state variables
    if is_new_conversation:
        print(f"[DEBUG] New conversation detected, resetting state for query: '{query[:50]}...'")
        return {
            "query": query,
            "waiting_for_leader": False,
            "original_query": "",
            "detected_leader": "",
            "tone_profile": "",
            "is_social_media": False,
            "grade": "yes",
            "feedback_collected": False,
            "waiting_for_feedback": False
        }
    else:
        print(f"[DEBUG] Continuing conversation with query: '{query[:50]}...'")
        return {"query": query}

def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents with feedback-enhanced scoring."""
    query = state["query"]
    
    # Get feedback storage for document scoring enhancement
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
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
        sources_text = "\n\n📚 **Sources:**\n" + "\n".join([f"- [View Document]({source})" for source in sources]) if sources else ""
        
        # Analyze query complexity to determine response approach
        is_analytical = any(word in query.lower() for word in ["analyze", "compare", "evaluate", "assess", "examples", "distinct"])
        is_actionable = any(word in query.lower() for word in ["how to", "steps", "implement", "strategy", "plan"])
        
        response_structure = "analytical" if is_analytical else "actionable" if is_actionable else "informational"
        
        prompt = f"""You are {detected_leader.upper()}, responding with your authentic voice and expertise. This is a {response_structure} query requiring a comprehensive, value-driven response.

🎯 YOUR VOICE & PERSPECTIVE ({detected_leader.upper()}):
{tone_profile}

📋 QUERY TO ADDRESS:
{query}

📊 KNOWLEDGE BASE CONTENT:
{context}

🏗️ RESPONSE FRAMEWORK:

1. **OPENING** (Establish Authority):
   - Acknowledge the question with {detected_leader}'s signature style
   - Preview the value you'll provide
   - Use {detected_leader}'s characteristic opening phrases

2. **CORE CONTENT** (Deliver Value):
   {"- Provide 3+ specific, distinct examples with clear explanations" if is_analytical else "- Give step-by-step actionable guidance" if is_actionable else "- Share comprehensive insights with practical applications"}
   - Use concrete details from the knowledge base
   - Include relevant data, statistics, or research findings
   - Frame everything through {detected_leader}'s unique perspective
   - Make connections to broader themes or implications

3. **ENGAGEMENT** (Drive Action):
   - Synthesize key takeaways
   - Provide actionable next steps or thought-provoking insights
   - End with {detected_leader}'s motivational style
   - Include relevant sources for credibility

📝 QUALITY STANDARDS:
- Every point must be substantiated by the knowledge base content
- Use specific examples, not generic statements  
- Maintain {detected_leader}'s authentic voice throughout
- Ensure practical applicability of insights
- Include precise details and avoid vague generalizations

{sources_text}

🎤 **Your Response as {detected_leader.upper()}:**"""
    else:
        prompt = f"""You are {detected_leader.upper()}, maintaining your authentic voice even when knowledge is limited.

🎯 YOUR VOICE ({detected_leader.upper()}):
{tone_profile}

❌ **Knowledge Gap Identified**
The available information doesn't contain sufficient relevant content to properly address this query: "{query}"

🗣️ **Your Response as {detected_leader.upper()}:**
Acknowledge the limitation authentically in your voice, explain what type of information would be needed, and offer alternative value or next steps that align with your leadership style. Maintain your characteristic tone while being transparent about the knowledge gap."""
    
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
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
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

def elicit_leader_and_tone(state: RAGState) -> RAGState:
    """Unified node: Detect leader, handle user selection if needed, and load tone profile."""
    query = state["query"]
    messages = state.get("messages", [])
    waiting_for_leader = state.get("waiting_for_leader", False)
    
    print(f"[DEBUG] elicit_leader_and_tone called with {len(messages)} messages, waiting_for_leader={waiting_for_leader}")
    
    # If we're already waiting for leader input, process the user's response
    if waiting_for_leader and len(messages) >= 2:
        # Look for a leader selection prompt in previous messages
        for i, msg in enumerate(messages[:-1]):
            if hasattr(msg, 'content') and "Choose Your Voice" in msg.content:
                # Found the prompt, now process the user's response
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    choice = last_message.content.strip().lower()
                elif isinstance(last_message, dict):
                    choice = last_message.get('content', '').strip().lower()
                else:
                    choice = str(last_message).strip().lower()
                
                print(f"[DEBUG] Processing leader choice: '{choice}'")
                
                # Map user choice to leader name
                leader_mapping = {"janelle": "janelle", "doreen": "doreen", "default": "default"}
                detected_leader = leader_mapping.get(choice, "default")
                
                # Load tone profile and use original query from state
                tone_profile = load_tone_profile(detected_leader)
                original_query = state.get("original_query", query)
                
                print(f"[DEBUG] Leader selected: {detected_leader}, proceeding with original query: '{original_query[:50]}...'")
                
                return {
                    "detected_leader": detected_leader,
                    "tone_profile": tone_profile,
                    "waiting_for_leader": False,
                    "query": original_query
                }
    
    # First-time processing: try to detect leader in query
    print(f"[DEBUG] First-time processing, detecting leader in query: '{query[:50]}...'")
    detection_prompt = f"""Does this query mention a specific leader name? Look for "Janelle" or "Doreen" anywhere in the text.

Query: {query}

If you see "Janelle" mentioned anywhere, respond with: janelle
If you see "Doreen" mentioned anywhere, respond with: doreen  
If neither name appears, respond with: none

Only respond with one word: janelle, doreen, or none"""
    
    detection_response = llm.invoke([HumanMessage(content=detection_prompt)])
    response_content = detection_response.content.strip().lower()
    
    print(f"[DEBUG] Leader detection result: '{response_content}' for query: '{query[:50]}...'")
    
    # Extract leader name from response (handle various formats)
    detected_leader = None
    if "janelle" in response_content:
        detected_leader = "janelle"
    elif "doreen" in response_content:
        detected_leader = "doreen"
    
    # If a leader was detected, proceed with that leader
    if detected_leader:
        tone_profile = load_tone_profile(detected_leader)
        print(f"[DEBUG] Leader '{detected_leader}' detected, proceeding with tone profile")
        return {
            "detected_leader": detected_leader,
            "tone_profile": tone_profile,
            "waiting_for_leader": False
        }
    
    # No leader detected - prompt user to choose (this is the safety net)
    print(f"[DEBUG] No leader detected (response: '{response_content}'), prompting user for selection")
    from langchain_core.messages import AIMessage
    
    leader_prompt = """
🎯 **Choose Your Voice**

I can respond in different leadership voices:

**Janelle** - Strategic business perspective with scaling mindset
**Doreen** - Relational, equity-focused approach  
**Default** - Professional, research-backed tone

Please reply with: **Janelle**, **Doreen**, or **Default**
    """.strip()
    
    result = {
        "messages": [AIMessage(content=leader_prompt)],
        "waiting_for_leader": True,
        "original_query": query
    }
    print(f"[DEBUG] Returning state with waiting_for_leader=True: {result.get('waiting_for_leader')}")
    return result


def collect_feedback(state: RAGState) -> RAGState:
    """Collect user feedback with rating buttons and optional text input."""
    print("[DEBUG] collect_feedback node called!")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from rag_2_0.feedback.feedback_storage import FeedbackStorage
        from langchain_core.messages import AIMessage
        
        storage = FeedbackStorage()
        response_id = state.get("response_id", "")
        
        # Simple feedback collection - just prompt once and that's it
        if response_id:
            feedback_prompt = """
📝 **Rate this response:**

Please rate from 1-5:
• **1** = Very Poor  
• **2** = Poor  
• **3** = Okay  
• **4** = Good  
• **5** = Excellent

*Optional: Add any specific feedback or suggestions*

(Note: In LangGraph Studio, feedback processing would be handled by a separate UI component)
            """.strip()
            
            feedback_message = AIMessage(content=feedback_prompt)
            current_messages = state.get("messages", [])
            
            print(f"[DEBUG] Prompting for feedback for response_id: {response_id}")
            
            return {
                "messages": current_messages + [feedback_message],
                "feedback_collected": True
            }
        
        # No response ID - skip feedback
        print("[DEBUG] No response_id found, skipping feedback")
        return state
        
    except Exception as e:
        print(f"[DEBUG] Error in collect_feedback: {e}")
        return state

def create_rag_graph():
    """Create the RAG workflow graph."""
    
    # Create workflow
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("extract_query", extract_query)
    workflow.add_node("detect_social_media", detect_social_media_request)
    workflow.add_node("elicit_leader_and_tone", elicit_leader_and_tone)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate_response)
    workflow.add_node("generate_social_media", generate_social_media_post)
    workflow.add_node("register_feedback", register_response_for_feedback)
    workflow.add_node("collect_feedback", collect_feedback)
    
    # Define workflow
    workflow.set_entry_point("extract_query")
    
    # Simple linear pipeline
    workflow.add_edge("extract_query", "detect_social_media")
    workflow.add_edge("detect_social_media", "elicit_leader_and_tone")
    
    # After elicit_leader_and_tone, check if we need to wait for user input
    workflow.add_conditional_edges(
        "elicit_leader_and_tone",
        lambda x: "END" if x.get("waiting_for_leader", False) else "retrieve",
        {
            "END": END,
            "retrieve": "retrieve"
        }
    )
    
    # Full RAG pipeline
    workflow.add_edge("retrieve", "grade_documents")
    
    # Branch based on social media flag
    workflow.add_conditional_edges(
        "grade_documents",
        lambda x: "generate_social_media" if x.get("is_social_media", False) else "generate",
        {
            "generate_social_media": "generate_social_media",
            "generate": "generate"
        }
    )
    
    # Both paths end with feedback registration
    workflow.add_edge("generate_social_media", "register_feedback")
    workflow.add_edge("generate", "register_feedback")
    
    # After registering, collect feedback
    workflow.add_edge("register_feedback", "collect_feedback")
    
    # End after feedback collection
    workflow.add_edge("collect_feedback", END)
    
    return workflow.compile()

# Create the graph for Studio
graph = create_rag_graph()