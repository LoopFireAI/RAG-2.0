"""
Simple Slack Bot API for Wells RAG 2.0
Lightweight FastAPI server for Slack integration
"""

import os
import sys
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
import logging
import hmac
import hashlib
import time
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_2_0.agents.rag_agent import create_rag_graph
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wells RAG Slack Bot", version="1.0.0")

# Initialize RAG system
rag_graph = create_rag_graph()

def verify_slack_signature(request_body: bytes, timestamp: str, signature: str) -> bool:
    """Verify that the request is coming from Slack"""
    slack_signing_secret = os.getenv("SLACK_SIGNING_SECRET", "")
    if not slack_signing_secret:
        logger.warning("SLACK_SIGNING_SECRET not set - skipping signature verification")
        return True
    
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False
    
    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    my_signature = 'v0=' + hmac.new(
        slack_signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(my_signature, signature)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Wells RAG Slack Bot is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "wells-rag-slack-bot"}

@app.post("/slack/events")
async def slack_events(request: Request):
    """Handle Slack Events API (app mentions)"""
    body = await request.body()
    headers = request.headers
    
    # Verify Slack signature
    timestamp = headers.get("x-slack-request-timestamp", "")
    signature = headers.get("x-slack-signature", "")
    
    if not verify_slack_signature(body, timestamp, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        data = json.loads(body.decode('utf-8'))
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return {"challenge": data.get("challenge")}
    
    # Handle app_mention events
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        
        if event.get("type") == "app_mention":
            user_message = event.get("text", "").strip()
            user = event.get("user")
            
            # Remove bot mention from message
            user_message = " ".join([word for word in user_message.split() if not word.startswith("<@")])
            
            if user_message:
                try:
                    response = await process_rag_query(user_message, user)
                    logger.info(f"Generated response for user {user}: {response[:100]}...")
                    # Note: You'd use Slack Web API here to send response back to channel
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
    
    return {"status": "ok"}

@app.post("/slack/commands/wells")
async def wells_slash_command(request: Request):
    """Handle /wells slash command"""
    body = await request.body()
    headers = request.headers
    
    # Verify Slack signature
    timestamp = headers.get("x-slack-request-timestamp", "")
    signature = headers.get("x-slack-signature", "")
    
    if not verify_slack_signature(body, timestamp, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Parse form data
    form_data = {}
    for item in body.decode('utf-8').split('&'):
        if '=' in item:
            key, value = item.split('=', 1)
            form_data[key] = value.replace('+', ' ').replace('%20', ' ')
    
    user_query = form_data.get("text", "").strip()
    user_id = form_data.get("user_id", "")
    user_name = form_data.get("user_name", "")
    
    if not user_query:
        return {
            "response_type": "ephemeral",
            "text": "Please provide a query. Example: `/wells What is effective leadership?`"
        }
    
    try:
        # Process the query with RAG system
        response = await process_rag_query(user_query, user_id, user_name)
        
        return {
            "response_type": "in_channel",
            "text": f"*Question:* {user_query}\n\n*Answer:*\n{response}",
            "mrkdwn": True
        }
        
    except Exception as e:
        logger.error(f"Error processing slash command: {e}")
        return {
            "response_type": "ephemeral",
            "text": "Sorry, I encountered an error processing your request. Please try again."
        }

async def process_rag_query(query: str, user_id: str, user_name: str = "") -> str:
    """Process a query through the RAG system"""
    try:
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Run through RAG workflow
        result = rag_graph.invoke(initial_state)
        
        # Extract response
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                response = last_message.content
            else:
                response = str(last_message)
        else:
            response = "I couldn't generate a response for your query."
        
        # Log for analytics
        logger.info(f"Query from {user_id} ({user_name}): '{query[:50]}...' -> Response length: {len(response)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG processing: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 