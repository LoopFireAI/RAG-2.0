"""
Socket Mode Slack Bot for Wells RAG 2.0
Uses WebSocket connection - no public URL needed!
"""

import os
import sys
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_2_0.agents.rag_agent import create_rag_graph
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Slack app with Socket Mode
app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

# Initialize RAG system
rag_graph = create_rag_graph()

# Get bot user ID for feedback validation
BOT_USER_ID = None

def process_rag_query(query: str, user_id: str, user_name: str = "") -> str:
    """Process a query through the RAG system"""
    try:
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Run through RAG workflow
        result = rag_graph.invoke(initial_state)
        
        # Extract the actual response (not the feedback prompt)
        response = ""
        if result.get("messages") and len(result["messages"]) >= 2:
            # The workflow adds feedback prompt as last message, so get the second-to-last
            response_message = result["messages"][-2]
            if hasattr(response_message, 'content'):
                response = response_message.content
            else:
                response = str(response_message)
        elif result.get("messages"):
            # Fallback to last message if only one exists
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
                # Check if it's a feedback prompt and skip it
                if "Rate this response" in content or "📝" in content:
                    response = "I couldn't generate a proper response for your query."
                else:
                    response = content
            else:
                response = str(last_message)
        else:
            response = "I couldn't generate a response for your query."
        
        # Add sources at the bottom for oversight and auditing
        try:
            # Log what we have in result for debugging
            logger.info(f"Result keys: {list(result.keys())}")
            logger.info(f"Retrieved docs metadata available: {bool(result.get('retrieved_docs_metadata'))}")
            logger.info(f"Sources available: {bool(result.get('sources'))}")
            
            if result.get("retrieved_docs_metadata"):
                docs_metadata = result["retrieved_docs_metadata"]
                logger.info(f"Found {len(docs_metadata)} doc metadata entries")
                
                # Log what's in the metadata for debugging
                for i, doc in enumerate(docs_metadata[:1]):  # Just log first one
                    logger.info(f"Doc {i} metadata keys: {list(doc.keys())}")
                    logger.info(f"Doc {i} title: {doc.get('title', 'No title')}")
                    logger.info(f"Doc {i} source: {doc.get('source', 'No source')}")
                
                # Enhanced source formatting for Slack
                response += f"\n\n---\n📚 **Sources** (for verification):"
                for i, doc in enumerate(docs_metadata[:3]):  # Limit to 3
                    # Try multiple fields for source name
                    title = doc.get('title') or doc.get('source') or 'Unknown Source'
                    
                    # Clean up the title extensively
                    clean_title = (title
                                 .replace('.pdf', '')
                                 .replace('_', ' ')
                                 .replace('-', ' ')
                                 .replace('.docx', '')
                                 .replace('.txt', '')
                                 .strip())
                    
                    # Handle Google Drive file names better
                    if '/d/' in clean_title:
                        clean_title = "Research Document"
                    elif 'drive.google.com' in clean_title:
                        clean_title = "Leadership Research Paper"
                    
                    # Proper title case
                    clean_title = ' '.join(word.capitalize() for word in clean_title.split())
                    
                    # Truncate if too long
                    if len(clean_title) > 45:
                        clean_title = clean_title[:42] + "..."
                    
                    response += f"\n• {clean_title}"
            
            # Fallback: try to get sources from other places in result
            elif result.get("sources"):
                sources = result["sources"][:3]  # Limit to 3
                logger.info(f"Using fallback sources: {sources}")
                response += f"\n\n---\n📚 **Sources** (for verification):"
                for source in sources:
                    # Clean up source names
                    clean_source = source.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
                    if len(clean_source) > 50:
                        clean_source = clean_source[:47] + "..."
                    response += f"\n• {clean_source}"
            else:
                logger.warning("No sources found in result")
                        
        except Exception as e:
            logger.error(f"Error formatting sources: {e}")
            # Add basic source info if available
            if result.get("sources"):
                response += f"\n\n---\n📚 **Sources**: {len(result['sources'])} research documents"
        
        # Log for analytics
        logger.info(f"Query from {user_id} ({user_name}): '{query[:50]}...' -> Response length: {len(response)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG processing: {e}")
        return "Sorry, I encountered an error processing your request. Please try again."

@app.event("app_mention")
def handle_mention(event, say, client, ack):
    """Handle @mentions of the bot"""
    ack()
    logger.info(f"🎯 App mention handler triggered!")
    
    user_message = event.get("text", "").strip()
    user = event.get("user")
    channel = event.get("channel")
    ts = event.get("ts")
    
    logger.info(f"📝 Original message: '{user_message}'")
    logger.info(f"👤 User: {user}")
    
    # Remove bot mention from message
    user_message = " ".join([word for word in user_message.split() if not word.startswith("<@")])
    cleaned_message = user_message.strip()
    
    logger.info(f"🧹 Cleaned message: '{cleaned_message}'")
    
    if cleaned_message:
        # User asked a specific question
        try:
            # Add eyes reaction to show we're processing
            try:
                client.reactions_add(
                    channel=channel,
                    timestamp=ts,
                    name="eyes"
                )
                logger.info("👀 Added eyes reaction")
            except Exception as e:
                logger.warning(f"Could not add reaction: {e}")
            
            logger.info(f"Processing question from {user}: {cleaned_message[:50]}...")
            response = process_rag_query(cleaned_message, user)
            
            # Reply in thread
            say(
                text=response,
                thread_ts=ts  # This makes it reply in thread
            )
            
            # Just remove eyes reaction - let USER add checkmark to trigger feedback
            try:
                client.reactions_remove(
                    channel=channel,
                    timestamp=ts,
                    name="eyes"
                )
                logger.info("👀 Removed eyes reaction - ready for user feedback")
            except Exception as e:
                logger.warning(f"Could not remove reaction: {e}")
            
        except Exception as e:
            logger.error(f"Error processing mention: {e}")
            # Remove eyes reaction and add error reaction
            try:
                client.reactions_remove(
                    channel=channel,
                    timestamp=ts,
                    name="eyes"
                )
                client.reactions_add(
                    channel=channel,
                    timestamp=ts,
                    name="x"
                )
            except:
                pass
            
            say(
                text="Sorry, I encountered an error processing your request. Please try again.",
                thread_ts=ts
            )
    else:
        # User just mentioned the bot without a question
        logger.info(f"User {user} mentioned bot without question - sending greeting")
        greeting = """👋 Hi there! I'm your Wells Leadership Research assistant.

I can help you explore insights from our extensive collection of leadership research papers. Just ask me questions like:
• "What makes an effective leader?"
• "How do leaders build trust?"
• "What are the key leadership competencies?"
• "Tell me about transformational leadership"

What would you like to know about leadership? 🚀"""
        
        say(
            text=greeting,
            thread_ts=ts  # Reply in thread even for greetings
        )

@app.command("/wells")
def handle_wells_command(ack, command, respond):
    """Handle /wells slash command"""
    ack()
    
    user_query = command["text"].strip()
    user_id = command["user_id"]
    user_name = command["user_name"]
    
    # Special debug command
    if user_query.lower() in ["debug", "test", "status"]:
        try:
            # Get bot info and scopes
            auth_info = app.client.auth_test()
            
            debug_info = f"""🔧 **Bot Debug Info**
• Bot User ID: {auth_info.get('user_id', 'Unknown')}
• Bot Name: {auth_info.get('user', 'Unknown')}
• Team: {auth_info.get('team', 'Unknown')}
• App ID: {auth_info.get('app_id', 'Unknown')}

📋 **Testing Permissions:**
• Can read messages: ✅ (you're seeing this)
• Can write messages: ✅ (you're seeing this)

🧪 **Reaction Test:**
Try adding a ✅ checkmark reaction to this message to test feedback system!

⚠️ **If reactions don't work, the Slack app needs:**
• `reactions:read` scope
• `reactions:write` scope  
• `reaction_added` event subscription

Current log will show if reaction events are received."""
            
            respond({
                "response_type": "in_channel",
                "text": debug_info,
                "mrkdwn": True
            })
            return
            
        except Exception as e:
            logger.error(f"Error in debug command: {e}")
            respond({
                "response_type": "ephemeral",
                "text": f"Debug command failed: {e}"
            })
            return
    
    if not user_query:
        respond({
            "response_type": "ephemeral",
            "text": "Please provide a query. Example: `/wells What is effective leadership?` or `/wells debug` for diagnostics"
        })
        return
    
    try:
        logger.info(f"Processing /wells command from {user_name}: {user_query[:50]}...")
        response = process_rag_query(user_query, user_id, user_name)
        
        respond({
            "response_type": "in_channel",
            "text": f"*Question:* {user_query}\n\n*Answer:*\n{response}",
            "mrkdwn": True
        })
        
    except Exception as e:
        logger.error(f"Error processing slash command: {e}")
        respond({
            "response_type": "ephemeral",
            "text": "Sorry, I encountered an error processing your request. Please try again."
        })

@app.event("reaction_added")
def handle_reaction_added(event, say, client, ack):
    """Handle when users react to bot messages"""
    ack()
    logger.info(f"🔄 Reaction added event received: {event}")
    
    # Log ALL reaction details for debugging
    reaction = event.get("reaction")
    user = event.get("user")
    item = event.get("item", {})
    logger.info(f"👤 User: {user}")
    logger.info(f"😀 Reaction: {reaction}")
    logger.info(f"📧 Channel: {item.get('channel')}")
    logger.info(f"⏰ Message TS: {item.get('ts')}")
    
    # For debugging: accept ANY reaction for now, log what we get
    if reaction in ["white_check_mark", "heavy_check_mark", "check", "checkmark", "+1", "thumbsup"]:
        logger.info(f"✅ Processing feedback reaction: {reaction}")
    else:
        logger.info(f"⏭️ Ignoring reaction: {reaction} (not a feedback reaction)")
        return
    
    # Only respond to reactions on bot messages
    user_id = event.get("user")
    item = event.get("item", {})
    channel = item.get("channel")
    message_ts = item.get("ts")
    
    if not all([user_id, channel, message_ts]):
        return
    
    try:
        # Get bot user ID if not cached
        global BOT_USER_ID
        if not BOT_USER_ID:
            BOT_USER_ID = client.auth_test()["user_id"]
        
        # Get the original message to check if it's from the bot
        response = client.conversations_history(
            channel=channel,
            latest=message_ts,
            limit=1,
            inclusive=True
        )
        
        messages = response.get("messages", [])
        if not messages:
            return
            
        message = messages[0]
        # Check if the message is from our bot (correct logic)
        logger.info(f"Message user: {message.get('user')}, Bot ID: {BOT_USER_ID}, Bot field: {message.get('bot_id')}")
        
        if message.get("user") == BOT_USER_ID or message.get("bot_id"):
            logger.info(f"📝 User {user_id} reacted with checkmark to bot message, prompting for feedback")
            
            # Check if we already prompted for feedback on this message
            try:
                thread_messages = client.conversations_replies(
                    channel=channel,
                    ts=message_ts,
                    limit=10
                ).get("messages", [])
                
                # Skip if feedback was already requested
                for thread_msg in thread_messages:
                    if (thread_msg.get("user") == BOT_USER_ID and 
                        "rate this response" in thread_msg.get("text", "").lower()):
                        logger.info("Feedback already requested for this message, skipping")
                        return
            except Exception as e:
                logger.warning(f"Could not check thread messages: {e}")
                # Continue anyway
            
            # Create feedback prompt with interactive buttons
            feedback_blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "📝 *Thanks for the checkmark!* How would you rate this response?"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "1 ⭐"},
                            "value": "1",
                            "action_id": "feedback_rating_1"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "2 ⭐⭐"},
                            "value": "2",
                            "action_id": "feedback_rating_2"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "3 ⭐⭐⭐"},
                            "value": "3",
                            "action_id": "feedback_rating_3"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "4 ⭐⭐⭐⭐"},
                            "value": "4",
                            "action_id": "feedback_rating_4"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "5 ⭐⭐⭐⭐⭐"},
                            "value": "5",
                            "action_id": "feedback_rating_5"
                        }
                    ]
                },
                {
                    "type": "input",
                    "element": {
                        "type": "plain_text_input",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Any specific feedback or suggestions? (optional)"
                        },
                        "action_id": "feedback_text"
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Additional Comments"
                    },
                    "optional": True
                }
            ]
            
            # Send feedback prompt in thread
            say(
                text="Please rate this response:",
                blocks=feedback_blocks,
                thread_ts=message_ts
            )
            
    except Exception as e:
        logger.error(f"Error handling reaction: {e}")

@app.action("feedback_rating_1")
@app.action("feedback_rating_2") 
@app.action("feedback_rating_3")
@app.action("feedback_rating_4")
@app.action("feedback_rating_5")
def handle_feedback_rating(ack, body, client, respond):
    """Handle feedback rating button clicks"""
    ack()
    
    try:
        # Extract rating from action_id
        action_id = body["actions"][0]["action_id"]
        rating = int(action_id.split("_")[-1])
        user_id = body["user"]["id"]
        
        # Get any text feedback from the input
        text_feedback = ""
        state_values = body.get("state", {}).get("values", {})
        for block_id, block_values in state_values.items():
            if "feedback_text" in block_values:
                text_feedback = block_values["feedback_text"].get("value", "")
                break
        
        logger.info(f"📊 Received feedback: Rating={rating}, User={user_id}, Text='{text_feedback[:50]}...'")
        
        # Store feedback in the system
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from rag_2_0.feedback.feedback_storage import FeedbackStorage
            
            storage = FeedbackStorage()
            
            # Store the feedback in the database
            feedback_id = storage.store_feedback(
                response_id=f"slack_{user_id}_{body.get('action_ts', '')}",  # Simple correlation
                rating=rating,
                feedback_text=text_feedback or None,
                user_id=user_id,
                platform="slack"
            )
            
            logger.info(f"💾 Stored feedback with ID: {feedback_id}")
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
        
        # Update the message to show feedback was received
        star_display = "⭐" * rating
        response_text = f"✅ *Feedback received!*\n\n🌟 **Rating:** {rating}/5 {star_display}"
        
        if text_feedback:
            response_text += f"\n💬 **Comments:** {text_feedback}"
        
        response_text += "\n\n_Thank you for helping improve our responses!_ 🙏"
        
        # Replace the feedback prompt with confirmation
        respond({
            "text": response_text,
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn", 
                        "text": response_text
                    }
                }
            ],
            "replace_original": True
        })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        respond({
            "text": "❌ Sorry, there was an error processing your feedback. Please try again.",
            "replace_original": True
        })

@app.event("message")
def handle_message_events(body, logger):
    """Handle other message events (optional)"""
    logger.info(f"📩 Message event received: {body}")

@app.middleware
def log_request(logger, body, next):
    """Log all incoming requests for debugging"""
    logger.info(f"🔍 Incoming event: {body.get('type', 'unknown')} - {body}")
    next()

# Add comprehensive event debugging
@app.event({"type": "reaction_added"})
def debug_reaction_added(body, logger):
    """Catch ALL reaction_added events for debugging"""
    logger.info(f"🔍 RAW reaction_added event: {body}")

@app.event({"type": "reaction_removed"})
def debug_reaction_removed(body, logger):
    """Catch ALL reaction_removed events for debugging"""
    logger.info(f"🔍 RAW reaction_removed event: {body}")

# Catch ALL events for debugging
@app.event(".*")
def debug_all_events(body, logger):
    """Catch ALL events to see what's being received"""
    event_type = body.get("event", {}).get("type", "unknown")
    if event_type not in ["app_mention", "message"]:  # Don't spam common events
        logger.info(f"🔍 DEBUG: {event_type} event: {body}")

if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    logger.info("🤖 Starting Wells RAG Slack Bot (Socket Mode)")
    logger.info("✅ Socket Mode - No public URL needed!")
    logger.info("✅ Supports @mentions and /wells commands")
    logger.info("✅ Supports checkmark reaction feedback collection")
    logger.info("ℹ️  Required scopes: app_mentions:read, channels:history, chat:write, chat:write.public, commands, reactions:read, reactions:write")
    logger.info("ℹ️  Required events: app_mention, reaction_added")
    logger.info("⚠️  DEBUGGING MODE: Will log ALL events and reactions")
    logger.info("🚀 Bot is connecting to Slack...")
    
    # Test reaction permissions on startup
    try:
        from slack_bolt import App
        test_client = app.client
        # Try to get bot info to verify connection
        bot_info = test_client.auth_test()
        logger.info(f"✅ Bot authenticated as: {bot_info.get('user', 'Unknown')}")
        logger.info(f"✅ Bot user ID: {bot_info.get('user_id', 'Unknown')}")
        
        # Log current token scopes if available
        try:
            auth_response = test_client.auth_test()
            if 'url' in auth_response:
                logger.info("✅ Bot has web API access")
        except Exception as e:
            logger.warning(f"⚠️ Auth test issue: {e}")
            
    except Exception as e:
        logger.error(f"❌ Bot authentication failed: {e}")
    
    # Start the Socket Mode handler
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start() 