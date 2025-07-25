"""
Simple RAG Agent for LangGraph Studio.
"""

from typing import TypedDict, List, Annotated, Literal
import operator
import os
import logging
import sys

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_agent.log')
    ]
)
logger = logging.getLogger(__name__)

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
    is_acknowledgment: bool  # Track if message is a simple acknowledgment

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
    query = state["query"]
    
    # Handle different query formats from LangGraph Studio
    if isinstance(query, list):
        # Extract text from list of dicts like [{'type': 'text', 'text': '...'}]
        if query and isinstance(query[0], dict) and 'text' in query[0]:
            query = query[0]['text']
        else:
            query = " ".join(str(item) for item in query)
    
    query = str(query).lower()

    # Comprehensive keyword detection for social media posts
    social_media_keywords = [
        "tweet", "twitter", "post", "social media", "linkedin", "facebook",
        "instagram", "thread", "threads", "make a post", "create a post",
        "linkedin post", "share on", "caption", "social", "engagement",
        "hashtag", "viral", "content", "share this", "post about",
        "social media post", "write a post", "create content"
    ]

    # Multi-word phrase detection (handles "linkedin post" at end of sentence)
    social_media_phrases = [
        "linkedin post", "twitter post", "facebook post", "instagram post",
        "social media post", "make a post", "create a post", "write a post",
        "give me a post", "generate a post", "post for linkedin", "post for twitter"
    ]
    
    # Check for explicit social media keywords anywhere in query
    has_social_keywords = any(keyword in query for keyword in social_media_keywords)
    
    # Check for multi-word phrases that might be split across query
    has_social_phrases = any(phrase in query for phrase in social_media_phrases)
    
    # Additional pattern detection for requests ending with social media terms
    ends_with_social = query.strip().endswith(('post', 'linkedin post', 'tweet', 'content'))
    
    is_social_media = has_social_keywords or has_social_phrases or ends_with_social
    
    logger.info(f"Social media detection: query='{query}', detected={is_social_media}")
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
    """Generate a concise social media post."""
    import uuid
    import time

    start_time = time.time()
    query = state["query"]
    context = state["context"]
    grade = state.get("grade", "yes")
    sources = state.get("sources", [])  # Get sources from state
    detected_leader = state.get("detected_leader", "default")
    tone_profile = state.get("tone_profile", "Use a professional and helpful tone.")
    retrieved_docs_metadata = state.get("retrieved_docs_metadata", [])

    # Generate unique response ID for feedback correlation
    response_id = str(uuid.uuid4())

    if grade == "yes":
        prompt = f"""You are {detected_leader.upper()}, sharing a warm, encouraging social media post that feels like advice from a trusted mentor.

YOUR VOICE ({detected_leader.upper()}):
{tone_profile}

QUERY TO ADDRESS:
{query}

RESPONSE STRUCTURE REQUIRED:
Based on the query structure, you must provide a direct answer that addresses the specific question format. 
- If asked for "3 things" - provide exactly 3 specific points
- If asked about "balance" - focus specifically on work-life balance concepts
- If asked about "professionals get wrong" - identify specific mistakes or misconceptions

KNOWLEDGE BASE CONTENT:
{context}

CRITICAL REQUIREMENTS:
- DO NOT start with disclaimers or limitations
- DO NOT use generic corporate language
- DO NOT provide vague or surface-level insights
- MUST extract specific, actionable insights from the knowledge base
- MUST create engaging, authentic content that provides real value
- MUST directly answer the specific question asked
- If the query asks for "3 things" or numbered items, provide exactly that structure
- If the retrieved content doesn't directly address the question, synthesize relevant insights that do
- MUST only have hashtags at the very end of the post. 

SOCIAL MEDIA POST REQUIREMENTS:
- Respond in short paragraphs. Please don't use bullet points in responses.
- Make sure the response is both detailed and concise. Use a call to action at the end of our captions inviting the audience to follow, read, and engage with future material.
- Call to action preferences: 
Awareness & Dialogue 
    - Pause and name the work that often goes unnoticed. 
    - Acknowledge the invisible, amplify the impact. 
    - Bring this to your next team meeting. 
    - Share this with someone doing the work no one talks about.  
Explore & Learn More 
    - Reframe what counts as work. Start here. 
    - Interested in learning more? Read the article here. 
    - Follow along for more conversations that shift perspective. 
    - The work doesn’t stop here, keep reading! 
Action & Impact 
    - Support the work behind the work. 
    - Ready to make change at work? Let’s talk.

- Use hashtags at the end of the post. 

Example of LinkedIn post: 
"What if the future of leadership isn’t about knowing all the answers, but asking better questions?

As AI reshapes the workplace, Ashley Lee challenges us to lead with intention, not urgency. In her latest piece, Shaping the Future of Work: Strategic Leadership in the Age of AI, she reminds us that technology doesn’t replace human connection, it requires more of it.

From addressing bias and protecting data to supporting teams through uncertainty, this article explores how thoughtful, people-first leadership can turn AI from a disruption into an opportunity.
“AI is not here to take away what makes us human. It’s here to help us become even better at it.”
Read the full article to explore how curiosity, care, and clarity can guide us forward.
hashtag#Leadership hashtag#FutureOfWork hashtag#AI hashtag#HumanCenteredLeadership"

Example of LinkedIn post for "3 things" query: 
"Three misconceptions about work-life balance that hold professionals back:

First, thinking balance means equal time allocation. Balance isn't about perfect 50-50 splits but about intentional choices that align with your values and current season of life.

Second, believing you need to achieve balance alone. The most successful professionals build systems and seek support rather than trying to juggle everything solo.

Third, treating balance as a destination rather than a dynamic practice. What works today might not work next month, and that's completely normal.

True balance comes from self-awareness, not self-sacrifice. What's one misconception you've let go of?

hashtag#WorkLifeBalance hashtag#ProfessionalDevelopment hashtag#Leadership"

Example of Instagram post: 
"Why do so many of us struggle to rest without guilt?
In this week’s question, Dr. Janelle Wells unpacks what drives our discomfort with downtime and why this mindset might be working against us.
Follow WellsQuest for thoughtful conversations that help reframe how we work and live.
#WellsQuest #QuestionOfTheWeek #DrJanelleWells #WellbeingAtWork #WorkRedefined"

Platform specific guidelines: 
LinkedIn: Thought leadership and professional storytelling with a focus on workplace equity or insights. Posts are reflective, research-backed, and cleanly formatted.  
 

X (Twitter): Concise, high-impact, and timely. We use bold statements or thred to amplify insights. Least engagement and reach with this platform and need to get more active in sharing industry insights and conversations. 
 

Instagram: Visual-first and emotionally resonant/warm. We use reels, carousels, quotes, and definition graphics to educate and connect. Captions are warm, concise, and reflective. 
  

Newsletters: Curated, slower-paced content with personal insights, key updates, and invitations to engage more deeply with WellsQuest’s work.  
  

Other (Facebook): Community-forward and personal tone, very similar to Instagram.

FORMAT:
Create a single, cohesive post (not sections) that flows naturally while incorporating these elements.

**Your Social Media Post as {detected_leader.upper()}:**"""
    else:
        prompt = f"""You are {detected_leader.upper()}, creating a social media post about knowledge limitations.

YOUR VOICE ({detected_leader.upper()}):
{tone_profile}

**Knowledge Gap for Query:** "{query}"

Create a brief, authentic social media post acknowledging this limitation while offering value in your characteristic style. Keep it under 100 words."""

    # Generate the main response
    response = llm.invoke([HumanMessage(content=prompt)])
    response_content = response.content

    # Add sources to the response content if available (compact for social media)
    if grade == "yes" and retrieved_docs_metadata:
        try:
            from rag_2_0.utils.source_formatter import SourceFormatter
            formatter = SourceFormatter()
            sources_formatted = formatter.format_sources_compact(retrieved_docs_metadata)
            if sources_formatted:
                response_content += sources_formatted
                logger.info(f"Added sources to social media post: {len(retrieved_docs_metadata)} sources")
            else:
                # Compact fallback for social media
                response_content += f"\n\nBased on {len(retrieved_docs_metadata)} research studies"
                logger.warning("Used fallback source formatting for social media")
        except Exception as e:
            logger.error(f"Error adding sources to social media post: {e}")
            # Simple fallback
            response_content += f"\n\nBased on research"

    # Create the response message with sources included
    from langchain_core.messages import AIMessage
    response_with_sources = AIMessage(content=response_content)

    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)

    # Store response time for feedback tracking
    for doc_meta in retrieved_docs_metadata:
        doc_meta["response_time_ms"] = response_time_ms

    # IMPORTANT: Return ALL state information including sources
    return {
        "messages": [response_with_sources],  # Use the response with sources included
        "response_id": response_id,
        "feedback_collected": False,
        "sources": sources,  # Keep sources in state
        "retrieved_docs_metadata": retrieved_docs_metadata  # Keep metadata in state
    }

def grade_documents(state: RAGState) -> RAGState:
    """Grade the relevance of retrieved documents to the query."""
    query = state["query"]
    context = state["context"]

    # If no context or very limited context, default to "yes" to prevent blocking
    if not context or len(context.strip()) < 50:
        logger.warning(f"Limited context ({len(context)} chars), defaulting to 'yes' grade")
        return {"grade": "yes"}

    prompt = GRADE_PROMPT.format(question=query, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])

    # Extract the grade from the response - be more permissive
    grade = "yes" if "yes" in response.content.lower() else "no"
    
    # Log grading decision for debugging
    logger.info(f"Document grading: query='{query[:50]}...', grade={grade}")

    return {"grade": grade}

def is_acknowledgment_message(content: str) -> bool:
    """Detect if a message is a simple acknowledgment that doesn't need full RAG processing."""
    content = content.lower().strip()
    
    # Common acknowledgment patterns
    acknowledgment_patterns = [
        "thank you", "thanks", "thank u", "thx", "ty",
        "great", "awesome", "perfect", "excellent", "nice",
        "got it", "ok", "okay", "alright", "sounds good",
        "appreciate it", "helpful", "that helps", "makes sense",
        "good to know", "understood", "i see", "interesting",
        "cool", "sweet", "nice work", "well done"
    ]
    
    # Check if the entire message is just an acknowledgment (with some flexibility for punctuation)
    cleaned_content = content.strip('!.,?').strip()
    
    # Direct matches
    if cleaned_content in acknowledgment_patterns:
        return True
    
    # Pattern matches for short messages
    if len(cleaned_content) <= 30:  # Only check short messages to avoid false positives
        for pattern in acknowledgment_patterns:
            if cleaned_content.startswith(pattern) or cleaned_content.endswith(pattern):
                return True
    
    return False

def handle_acknowledgment(state: RAGState) -> RAGState:
    """Handle acknowledgment messages with simple responses."""
    import uuid
    from langchain_core.messages import AIMessage
    
    # Simple, varied acknowledgment responses
    responses = [
        "You're welcome! Feel free to ask if you need anything else.",
        "Glad I could help! Let me know if you have other questions.",
        "Happy to assist! Reach out anytime.",
        "You're welcome! I'm here whenever you need support."
    ]
    
    # Pick a response (could be random, but keeping it simple)
    response = responses[0]
    
    # Create response message
    response_message = AIMessage(content=response)
    
    logger.info(f"Handled acknowledgment with simple response: '{response}'")
    
    return {
        "messages": [response_message],
        "response_id": str(uuid.uuid4()),
        "feedback_collected": True,  # Skip feedback for acknowledgments
        "is_acknowledgment": True
    }

def extract_query(state: RAGState) -> RAGState:
    """Extract query from messages and reset state for new conversations."""
    messages = state.get("messages", [])

    # Check if this is a new conversation (only one message, which should be the user's query)
    is_new_conversation = len(messages) == 1

    if messages:
        # Look for the actual query (not just voice selections)
        query = None
        
        # Process messages to find the real query and voice selections
        for message in messages:
            if hasattr(message, 'content'):
                content = message.content.strip()
            elif isinstance(message, dict):
                content = message.get('content', '').strip()
            else:
                content = str(message).strip()
            
            # Skip voice selection responses
            if content.lower() in ["janelle", "doreen", "default"]:
                continue
                
            # Skip bot messages asking for voice
            if "Choose Your Voice" in content:
                continue
                
            # This should be the actual query
            if content and len(content) > 10:  # Reasonable query length
                query = content
                break
        
        # If no proper query found, use the last message
        if not query:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                query = last_message.content
            elif isinstance(last_message, dict):
                query = last_message.get('content', '')
            else:
                query = str(last_message)
    else:
        query = "What is machine learning?"

    # Check if this is an acknowledgment message
    is_ack = is_acknowledgment_message(query)
    
    # For new conversations, reset all state variables
    is_ack = is_acknowledgment_message(query)
    if is_new_conversation:
        logger.info(f"New conversation detected, resetting state for query: '{query[:50]}...'")
        return {
            "query": query,
            "waiting_for_leader": False,
            "original_query": "",
            "detected_leader": "",
            "tone_profile": "",
            "is_social_media": False,
            "grade": "yes",
            "feedback_collected": False,
            "waiting_for_feedback": False,
            "is_acknowledgment": is_ack
        }
    else:
        logger.info(f"Continuing conversation with query: '{query[:50]}...'")
        return {"query": query, "is_acknowledgment": is_ack}

def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents with feedback-enhanced scoring and fallback logic."""
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

    # Use LangChain Chroma similarity search with expanded retrieval
    top_k = int(os.getenv("TOP_K", 5))  # Increase default from 3 to 5
    results = vector_store.similarity_search(query, k=top_k)
    
    # Extract key topic from query for targeted search
    query_lower = query.lower()
    key_topics = []
    if "balance" in query_lower:
        key_topics.append("work-life balance")
    if "leadership" in query_lower:
        key_topics.append("leadership")
    if "professional" in query_lower:
        key_topics.append("professional development")
    
    # If we have key topics and initial results are limited, do targeted search
    if key_topics and len(results) < top_k:
        for topic in key_topics:
            topic_results = vector_store.similarity_search(topic, k=top_k//2)
            seen_content = {doc.page_content for doc in results}
            for doc in topic_results:
                if doc.page_content not in seen_content and len(results) < top_k:
                    results.append(doc)
                    seen_content.add(doc.page_content)
        logger.info(f"Added targeted search results for topics: {key_topics}")
    
    # Fallback search with relaxed terms if initial results are limited
    if len(results) < top_k // 2:  # If we get less than half expected results
        # Extract key terms from query and try broader search
        query_terms = query.lower().split()
        important_terms = [term for term in query_terms if len(term) > 3 and term not in ['what', 'how', 'why', 'when', 'where', 'give', 'make', 'create']]
        
        if important_terms:
            # Try search with just key terms
            broader_query = ' '.join(important_terms[:3])  # Use top 3 key terms
            logger.info(f"Fallback search with broader query: '{broader_query}'")
            additional_results = vector_store.similarity_search(broader_query, k=top_k)
            
            # Merge results, avoiding duplicates
            seen_content = {doc.page_content for doc in results}
            for doc in additional_results:
                if doc.page_content not in seen_content and len(results) < top_k:
                    results.append(doc)
                    seen_content.add(doc.page_content)

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

    # Extract document content and format with clear separators for better LLM processing
    documents = [doc.page_content for doc in results]
    formatted_docs = []
    for i, doc in enumerate(results, 1):
        doc_content = doc.page_content.strip()
        # Add document separator with metadata if available
        doc_title = doc.metadata.get('title', f'Document {i}')
        formatted_doc = f"=== {doc_title} ===\n{doc_content}"
        formatted_docs.append(formatted_doc)
    
    context = "\n\n".join(formatted_docs)

    # Load document titles from JSON
    import json
    import os
    try:
        titles_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'document_titles.json')
        with open(titles_file, 'r', encoding='utf-8') as f:
            title_mapping = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load document titles: {e}")
        title_mapping = {}

    # Extract sources and metadata for feedback
    sources = []
    retrieved_docs_metadata = []
    for doc in results:
        if 'source' in doc.metadata:
            sources.append(doc.metadata['source'])

        # Get clean title from mapping or fallback to source
        source_url = doc.metadata.get('source', '')
        clean_title = title_mapping.get(source_url, source_url)

        # Store full metadata for feedback correlation
        retrieved_docs_metadata.append({
            'id': doc.metadata.get('id', doc.metadata.get('source', '')),
            'title': clean_title,
            'source': doc.metadata.get('source', ''),
            'metadata': doc.metadata,
            'content_preview': doc.page_content[:200]
        })

    # Debug log statements
    logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
    for i, doc in enumerate(results):
        try:
            logger.debug(f"Doc {i+1} content (first 100 chars): {doc.page_content[:100]}")
            if 'source' in doc.metadata:
                logger.debug(f"Doc {i+1} source: {doc.metadata['source']}")
            if 'feedback_boost' in doc.metadata:
                logger.debug(f"Doc {i+1} feedback boost: {doc.metadata['feedback_boost']}")
        except Exception as e:
            logger.error(f"Error reading content of doc {i+1}: {e}")

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
    retrieved_docs_metadata = state.get("retrieved_docs_metadata", [])

    # Generate unique response ID for feedback correlation
    response_id = str(uuid.uuid4())

    # Only use context if it was graded as relevant
    if grade == "yes":
        # Import and use the improved source formatter
        from rag_2_0.utils.source_formatter import SourceFormatter
        formatter = SourceFormatter()
        
        # Use the retrieved docs metadata for better formatting
        # sources_text = formatter.format_sources_section(retrieved_docs_metadata)  # Unused variable

        # Analyze query complexity to determine response approach
        is_analytical = any(word in query.lower() for word in ["analyze", "compare", "evaluate", "assess", "examples", "distinct"])
        is_actionable = any(word in query.lower() for word in ["how to", "steps", "implement", "strategy", "plan"])

        response_structure = "analytical" if is_analytical else "actionable" if is_actionable else "informational"

        prompt = f"""You are {detected_leader.upper()}, having a warm conversation with a trusted colleague who needs practical guidance. This is NOT an academic presentation - it's a supportive, wise conversation.

YOUR VOICE & PERSPECTIVE ({detected_leader.upper()}):
{tone_profile}

QUERY TO ADDRESS:
{query}

KNOWLEDGE BASE CONTENT:
{context}

RESPONSE FRAMEWORK:

MANDATORY STORYTELLING APPROACH:
1. Start with a relatable question or story that validates their experience
2. Use metaphors, analogies, or everyday examples to explain concepts  
3. Share specific workplace scenarios readers can immediately recognize
4. Bridge research to practice through storytelling, not bullet points
5. End with genuine encouragement and community connection

REQUIRED ELEMENTS:
- Use "you" language to speak directly to the reader
- Include at least one metaphor or analogy
- Provide specific, actionable steps they can try tomorrow
- Reference research naturally within stories, not as separate citations
- Acknowledge the emotional reality of their workplace challenges
- Validate their struggle before offering solutions
- Use warm, encouraging language throughout

CONVERSATION FLOW:
- Hook: Relatable question or scenario
- Heart: Acknowledge their emotional reality with warmth
- Help: Practical guidance through storytelling
- Hope: Encouraging next steps and community connection

QUALITY STANDARDS:
- SYNTHESIZE information from the knowledge base - don't just quote or excerpt
- Transform raw research into {detected_leader}'s authentic insights and perspective
- Use specific examples and data points, but frame them in your voice
- Maintain {detected_leader}'s authentic voice throughout
- Ensure practical applicability of insights
- Create original value-added commentary, not just information regurgitation

**Your Response as {detected_leader.upper()}:**"""
    else:
        prompt = f"""You are {detected_leader.upper()}, maintaining your authentic voice even when knowledge is limited.

YOUR VOICE ({detected_leader.upper()}):
{tone_profile}

**Knowledge Gap Identified**
The available information doesn't contain sufficient relevant content to properly address this query: "{query}"

**Your Response as {detected_leader.upper()}:**
Acknowledge the limitation authentically in your voice, explain what type of information would be needed, and offer alternative value or next steps that align with your leadership style. Maintain your characteristic tone while being transparent about the knowledge gap."""

    # Generate the main response
    response = llm.invoke([HumanMessage(content=prompt)])
    response_content = response.content

    # Add sources to the response content if available
    if grade == "yes" and retrieved_docs_metadata:
        try:
            from rag_2_0.utils.source_formatter import SourceFormatter
            formatter = SourceFormatter()
            sources_formatted = formatter.format_sources_compact(retrieved_docs_metadata)
            if sources_formatted:
                response_content += sources_formatted
                logger.info(f"Added sources to response content: {len(retrieved_docs_metadata)} sources")
            else:
                # Fallback source formatting
                response_content += f"\n\n**Sources:** {len(retrieved_docs_metadata)} research documents"
                logger.warning("Used fallback source formatting")
        except Exception as e:
            logger.error(f"Error adding sources to response: {e}")
            # Simple fallback
            response_content += f"\n\n**Sources:** {len(retrieved_docs_metadata)} research documents"

    # Create the response message with sources included
    from langchain_core.messages import AIMessage
    response_with_sources = AIMessage(content=response_content)

    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)

    # Log token usage if available
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        token_usage = response.response_metadata['token_usage']
        logger.info(f"Token usage - Input: {token_usage.get('prompt_tokens', 0)}, Output: {token_usage.get('completion_tokens', 0)}, Total: {token_usage.get('total_tokens', 0)}")

    # Store response time for feedback tracking
    for doc_meta in retrieved_docs_metadata:
        doc_meta["response_time_ms"] = response_time_ms

    # IMPORTANT: Return ALL state information including sources
    return {
        "messages": [response_with_sources],  # Use the response with sources included
        "response_id": response_id,
        "feedback_collected": False,
        "sources": sources,  # Keep sources in state
        "retrieved_docs_metadata": retrieved_docs_metadata  # Keep metadata in state
    }

def register_response_for_feedback(state: RAGState) -> RAGState:
    """Register response with feedback collector for potential feedback collection."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from rag_2_0.feedback.feedback_collector import FeedbackCollector
        from rag_2_0.feedback.feedback_storage import FeedbackStorage

        logger.debug("Attempting to register response for feedback...")

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

        logger.debug(f"Response ID: {response_id}")
        logger.debug(f"Query: {query[:50]}...")
        logger.debug(f"Response content length: {len(response_content)}")
        logger.debug(f"Retrieved docs: {len(retrieved_docs)}")

        if response_id and response_content and query:
            registered_id = collector.register_response(
                query=query,
                response=response_content,
                retrieved_docs=retrieved_docs,
                persona=state.get("detected_leader", "default"),
                response_time_ms=retrieved_docs[0].get("response_time_ms", 0) if retrieved_docs else 0,
                response_id=response_id  # Use the existing response_id
            )
            logger.debug(f"Successfully registered response with ID: {registered_id}")
        else:
            logger.debug(f"Failed to register: missing data - response_id={bool(response_id)}, content={bool(response_content)}, query={bool(query)}")

    except ImportError:
        logger.debug("Feedback system not available (ImportError)")
    except Exception as e:
        logger.error(f"Error registering response for feedback: {e}")
        import traceback
        traceback.print_exc()

    return state

def elicit_leader_and_tone(state: RAGState) -> RAGState:
    """Unified node: Detect leader, handle user selection if needed, and load tone profile."""
    query = state["query"]
    messages = state.get("messages", [])
    waiting_for_leader = state.get("waiting_for_leader", False)

    logger.debug(f"elicit_leader_and_tone called with {len(messages)} messages, waiting_for_leader={waiting_for_leader}")

    # Check if the most recent message is a voice selection response
    if len(messages) >= 1:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            last_content = last_message.content.strip().lower()
        elif isinstance(last_message, dict):
            last_content = last_message.get('content', '').strip().lower()
        else:
            last_content = str(last_message).strip().lower()
        
        # Check if this looks like a voice selection
        if last_content in ["janelle", "doreen", "default"]:
            logger.debug(f"Processing voice selection: '{last_content}'")
            
            # Map user choice to leader name
            detected_leader = last_content
            tone_profile = load_tone_profile(detected_leader)
            
            # Use original query if available, otherwise use current query
            original_query = state.get("original_query", query)
            
            logger.debug(f"Leader selected: {detected_leader}, proceeding with query: '{original_query[:50]}...'")
            
            return {
                "detected_leader": detected_leader,
                "tone_profile": tone_profile,
                "waiting_for_leader": False,
                "query": original_query
            }
    
    # Legacy handling: If we're explicitly waiting for leader input, process the user's response
    if waiting_for_leader and len(messages) >= 2:
        # Look for a leader selection prompt in previous messages
        for msg in messages[:-1]:
            if hasattr(msg, 'content') and "Choose Your Voice" in msg.content:
                # Found the prompt, now process the user's response
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    choice = last_message.content
                elif isinstance(last_message, dict):
                    choice = last_message.get('content', '')
                else:
                    choice = str(last_message)
                
                # Handle LangGraph Studio format: [{'type': 'text', 'text': '...'}]
                if isinstance(choice, list):
                    if choice and isinstance(choice[0], dict) and 'text' in choice[0]:
                        choice = choice[0]['text']
                    else:
                        choice = " ".join(str(item) for item in choice)
                
                choice = str(choice).strip().lower()

                logger.debug(f"Processing leader choice: '{choice}'")

                # Map user choice to leader name (case-insensitive)
                leader_mapping = {"janelle": "janelle", "doreen": "doreen", "default": "default"}
                detected_leader = leader_mapping.get(choice, "default")

                # Load tone profile and use original query from state
                tone_profile = load_tone_profile(detected_leader)
                original_query = state.get("original_query", query)

                logger.debug(f"Leader selected: {detected_leader}, proceeding with original query: '{original_query[:50]}...'")

                return {
                    "detected_leader": detected_leader,
                    "tone_profile": tone_profile,
                    "waiting_for_leader": False,
                    "query": original_query
                }

    # First-time processing: try to detect leader in query
    logger.debug(f"First-time processing, detecting leader in query: '{query[:50]}...'")
    detection_prompt = f"""Does this query mention a specific leader name? Look for "Janelle" or "Doreen" anywhere in the text.

Query: {query}

If you see "Janelle" mentioned anywhere, respond with: janelle
If you see "Doreen" mentioned anywhere, respond with: doreen  
If neither name appears, respond with: none

Only respond with one word: janelle, doreen, or none"""

    detection_response = llm.invoke([HumanMessage(content=detection_prompt)])
    response_content = detection_response.content.strip().lower()

    logger.debug(f"Leader detection result: '{response_content}' for query: '{query[:50]}...'")

    # Extract leader name from response (handle various formats)
    detected_leader = None
    if "janelle" in response_content:
        detected_leader = "janelle"
    elif "doreen" in response_content:
        detected_leader = "doreen"

    # If a leader was detected, proceed with that leader
    if detected_leader:
        tone_profile = load_tone_profile(detected_leader)
        logger.debug(f"Leader '{detected_leader}' detected, proceeding with tone profile")
        return {
            "detected_leader": detected_leader,
            "tone_profile": tone_profile,
            "waiting_for_leader": False
        }

    # No leader detected - prompt user to choose (this is the safety net)
    logger.debug(f"No leader detected (response: '{response_content}'), prompting user for selection")
    from langchain_core.messages import AIMessage

    leader_prompt = """
Choose the voice that you want me to use to write this:

Janelle
Doreen
Default

Please reply with: Janelle, Doreen, or Default
    """.strip()

    result = {
        "messages": [AIMessage(content=leader_prompt)],
        "waiting_for_leader": True,
        "original_query": query
    }
    logger.debug(f"Returning state with waiting_for_leader=True: {result.get('waiting_for_leader')}")
    return result


def collect_feedback(state: RAGState) -> RAGState:
    """Collect user feedback with rating buttons and optional text input."""
    logger.debug("collect_feedback node called!")

    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        # from rag_2_0.feedback.feedback_storage import FeedbackStorage  # Unused import
        from langchain_core.messages import AIMessage

        # storage = FeedbackStorage()  # Unused variable
        response_id = state.get("response_id", "")

        # Simple feedback collection - just prompt once and that's it
        if response_id:
            feedback_prompt = """
**Rate this response:**

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

            logger.debug(f"Prompting for feedback for response_id: {response_id}")

            return {
                "messages": current_messages + [feedback_message],
                "feedback_collected": True
            }

        # No response ID - skip feedback
        logger.debug("No response_id found, skipping feedback")
        return state

    except Exception as e:
        logger.error(f"Error in collect_feedback: {e}")
        return state

def create_rag_graph():
    """Create the RAG workflow graph."""

    # Create workflow
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("extract_query", extract_query)
    workflow.add_node("handle_acknowledgment", handle_acknowledgment)
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

    # Check for acknowledgments first
    workflow.add_conditional_edges(
        "extract_query",
        lambda x: "handle_acknowledgment" if x.get("is_acknowledgment", False) else "detect_social_media",
        {
            "handle_acknowledgment": "handle_acknowledgment",
            "detect_social_media": "detect_social_media"
        }
    )
    
    # Acknowledgments go straight to END
    workflow.add_edge("handle_acknowledgment", END)
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

    # Branch based on social media flag with debugging
    def routing_decision(state):
        is_social = state.get("is_social_media", False)
        decision = "generate_social_media" if is_social else "generate"
        logger.info(f"Routing decision: is_social_media={is_social}, route={decision}")
        return decision
    
    workflow.add_conditional_edges(
        "grade_documents",
        routing_decision,
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
