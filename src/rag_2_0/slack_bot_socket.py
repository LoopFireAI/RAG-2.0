"""
Socket Mode Slack Bot for Wells RAG 2.0
Uses WebSocket connection - no public URL needed!
"""

import os
import sys
import logging
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_2_0.agents.rag_agent import create_rag_graph
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Log token info (first 20 chars only for security)
bot_token = os.getenv("SLACK_BOT_TOKEN")
signing_secret = os.getenv("SLACK_SIGNING_SECRET")
logger.info(f"Loading SLACK_BOT_TOKEN: {bot_token[:20] if bot_token else 'NOT_SET'}...")
logger.info(f"Loading SLACK_SIGNING_SECRET: {signing_secret[:20] if signing_secret else 'NOT_SET'}...")

# Initialize Slack app with Socket Mode
app = App(
    token=bot_token,
    signing_secret=signing_secret
)

# Initialize RAG system
rag_graph = create_rag_graph()

# Get bot user ID for feedback validation
BOT_USER_ID = None

def clean_response_for_slack(response: str) -> str:
    """Clean up response formatting for better Slack presentation"""
    if not response:
        return response
    
    # Replace markdown bold with Slack-friendly formatting
    # Convert **text** to *text* (Slack's bold format)
    import re
    response = re.sub(r'\*\*(.*?)\*\*', r'*\1*', response)
    
    # Clean up excessive line breaks
    response = re.sub(r'\n{3,}', '\n\n', response)  # Max 2 consecutive line breaks
    
    # Improve sources section formatting for Slack
    # Match both the original and converted patterns
    sources_patterns = [
        r'📚 \*\*Sources:\*\*(.*?)$',  # Original pattern
        r'📚 \*Sources:\*(.*?)$'       # After ** conversion
    ]
    
    for pattern in sources_patterns:
        sources_match = re.search(pattern, response, re.DOTALL)
        if sources_match:
            sources_content = sources_match.group(1).strip()
            # Clean up the sources formatting
            sources_lines = [line.strip() for line in sources_content.split('•') if line.strip()]
            
            if sources_lines:
                clean_sources = "\n\n> *Sources:*"
                for source in sources_lines[:3]:  # Limit to 3 sources
                    # Remove extra formatting and clean up
                    clean_source = source.replace('*', '').strip()
                    if clean_source:
                        clean_sources += f"\n> • {clean_source}"
                
                # Replace the original sources section
                response = re.sub(pattern, clean_sources, response, flags=re.DOTALL)
                break  # Only process once
    
    # Ensure the response doesn't end with excessive spacing
    response = response.strip()
    
    return response

def process_rag_query(query: str, user_id: str, user_name: str = "") -> str:
    """Process a query through the RAG system"""
    try:
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Run through RAG workflow
        result = rag_graph.invoke(initial_state)
        
        # Extract the actual response (skip feedback prompts)
        response = ""
        if result.get("messages"):
            # Look for the main response (not feedback prompts)
            for message in reversed(result["messages"]):  # Start from the end
                if hasattr(message, 'content'):
                    content = message.content
                elif isinstance(message, dict):
                    content = message.get('content', '')
                else:
                    content = str(message)
                
                # Skip feedback prompts and find the actual response
                if (content and 
                    "Rate this response" not in content and 
                    "📝" not in content[:10] and  # Skip feedback emojis at start
                    "Choose Your Voice" not in content and
                    len(content) > 100):  # Actual responses should be substantial
                    response = content
                    break
            
            # Fallback: if no good response found, use the first substantial message
            if not response and len(result["messages"]) > 1:
                first_msg = result["messages"][1] if len(result["messages"]) > 1 else result["messages"][0]
                if hasattr(first_msg, 'content'):
                    response = first_msg.content
                elif isinstance(first_msg, dict):
                    response = first_msg.get('content', '')
                else:
                    response = str(first_msg)
        
        # Final fallback
        if not response:
            response = "I couldn't generate a response for your query."

        # Clean up formatting for Slack presentation
        response = clean_response_for_slack(response)
        
        # Log for analytics
        logger.info(f"Query from {user_id} ({user_name}): '{query[:50]}...' -> Response length: {len(response)}")
        logger.info(f"Sources included in response: {'Sources' in response}")
        
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
    
    # Special feedback test command
    if user_query.lower() in ["feedback", "test-feedback", "feedback-test"]:
        try:
            # Send a test message that should trigger feedback when reacted to
            test_response = respond({
                "response_type": "in_channel",
                "text": """🧪 **Feedback Test Message**
                
This is a test message from the Wells RAG bot. 

**To test feedback system:**
1. Add a ✅ checkmark reaction to this message
2. If working, you should see a feedback prompt with rating buttons
3. Check the logs for reaction events

**Expected behavior:**
✅ Reaction detected → Feedback prompt appears → Rating collected

If this doesn't work, check that the Slack app has:
• `reactions:read` scope
• `reactions:write` scope  
• `reaction_added` event subscription""",
                "mrkdwn": True
            })
            logger.info(f"📤 Sent feedback test message: {test_response}")
            return
            
        except Exception as e:
            logger.error(f"Error in feedback test command: {e}")
            respond({
                "response_type": "ephemeral",
                "text": f"Feedback test failed: {e}"
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
    logger.info(f"🔄 REACTION EVENT RECEIVED!")
    logger.info(f"🔍 Full event data: {event}")
    
    # Log ALL reaction details for debugging
    reaction = event.get("reaction")
    user = event.get("user")
    item = event.get("item", {})
    logger.info(f"👤 User: {user}")
    logger.info(f"😀 Reaction emoji name: '{reaction}'")
    logger.info(f"📧 Channel: {item.get('channel')}")
    logger.info(f"⏰ Message TS: {item.get('ts')}")
    
    # ENHANCED: Log every single reaction we receive to identify the correct name
    logger.info(f"🧪 TESTING: Is '{reaction}' a checkmark reaction?")
    
    # Expanded list of possible checkmark reaction names
    valid_reactions = [
        "white_check_mark", "heavy_check_mark", "check", "checkmark", 
        "+1", "thumbsup", "thumbs_up", "ballot_box_with_check",
        "white-check-mark", "heavy-check-mark", "check-mark",
        "tick", "approved", "done", "yes"
    ]
    
    if reaction in valid_reactions:
        logger.info(f"✅ Processing feedback reaction: {reaction}")
    else:
        logger.info(f"⏭️ Ignoring reaction: {reaction} (not a feedback reaction)")
        logger.info(f"🔍 Available valid reactions: {valid_reactions}")
        # STILL CONTINUE to check if it's on a bot message for debugging - we want to see all reactions
        
    # Only respond to reactions on bot messages
    user_id = event.get("user")
    item = event.get("item", {})
    channel = item.get("channel")
    message_ts = item.get("ts")
    
    if not all([user_id, channel, message_ts]):
        logger.warning(f"Missing required data: user_id={user_id}, channel={channel}, message_ts={message_ts}")
        return
    
    try:
        # Get bot user ID if not cached
        global BOT_USER_ID
        if not BOT_USER_ID:
            auth_response = client.auth_test()
            BOT_USER_ID = auth_response["user_id"]
            logger.info(f"🤖 Bot user ID set to: {BOT_USER_ID}")
        
        # Get the original message to check if it's from the bot
        response = client.conversations_history(
            channel=channel,
            latest=message_ts,
            limit=1,
            inclusive=True
        )
        
        messages = response.get("messages", [])
        if not messages:
            logger.warning(f"No message found for ts={message_ts}")
            return
            
        message = messages[0]
        message_user = message.get("user")
        message_bot_id = message.get("bot_id")
        
        logger.info(f"📧 Message details - User: {message_user}, Bot ID: {message_bot_id}, Our Bot ID: {BOT_USER_ID}")
        
        # Check if the message is from our bot 
        is_bot_message = (message_user == BOT_USER_ID) or bool(message_bot_id)
        
        if not is_bot_message:
            logger.info(f"🚫 Reaction not on bot message (message_user={message_user}, bot_id={message_bot_id})")
            return
            
        # Only proceed if it's a valid feedback reaction
        if reaction not in valid_reactions:
            logger.info(f"🚫 Reaction '{reaction}' not in valid feedback reactions")
            return
            
        logger.info(f"📝 User {user_id} reacted with '{reaction}' to bot message, prompting for feedback")
        
        # Check if we already prompted for feedback on this message
        try:
            thread_messages = client.conversations_replies(
                channel=channel,
                ts=message_ts,
                limit=20  # Increased limit to catch more thread messages
            ).get("messages", [])
            
            # Skip if feedback was already requested
            feedback_already_requested = False
            for thread_msg in thread_messages:
                msg_text = thread_msg.get("text", "").lower()
                if (thread_msg.get("user") == BOT_USER_ID and 
                    ("rate this response" in msg_text or "thanks for the checkmark" in msg_text)):
                    logger.info("Feedback already requested for this message, skipping")
                    feedback_already_requested = True
                    break
                    
            if feedback_already_requested:
                return
                
        except Exception as e:
            logger.warning(f"Could not check thread messages: {e}")
            # Continue anyway - better to potentially duplicate than miss feedback
        
        # Create feedback prompt with interactive buttons
        feedback_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"✨ *Thanks for the feedback!* How would you rate this response?"
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
                        "text": "Optional: Share specific feedback or suggestions..."
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
        try:
            feedback_response = say(
                text="Please rate this response:",
                blocks=feedback_blocks,
                thread_ts=message_ts
            )
            logger.info(f"✅ Feedback prompt sent successfully: {feedback_response}")
        except Exception as e:
            logger.error(f"❌ Failed to send feedback prompt: {e}")
            # Fallback to simple text message
            say(
                text=f"Thanks for the {reaction}! Please rate this response (1-5) and optionally provide feedback.",
                thread_ts=message_ts
            )
            
    except Exception as e:
        logger.error(f"Error handling reaction: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

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
        
        # Create a clean, professional confirmation message
        if text_feedback:
            response_text = f"✅ *Thank you for your feedback!*\n\n🌟 *Rating:* {rating}/5 {star_display}\n💬 *Comments:* {text_feedback}\n\n_Your feedback helps us improve our responses._"
        else:
            response_text = f"✅ *Thank you for your feedback!*\n\n🌟 *Rating:* {rating}/5 {star_display}\n\n_Your feedback helps us improve our responses._"
        
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

# Add a specific handler to log ALL incoming webhooks
@app.middleware
def log_all_requests(body, next, logger):
    """Log all incoming Slack events for debugging"""
    event_type = body.get("type", "unknown")
    if event_type == "event_callback":
        inner_event = body.get("event", {})
        inner_type = inner_event.get("type", "unknown")
        logger.info(f"🌐 Event callback received: {inner_type}")
        if inner_type == "reaction_added":
            logger.info(f"🎯 REACTION EVENT DETECTED: {inner_event}")
    else:
        logger.info(f"🌐 Request type: {event_type}")
    next()

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