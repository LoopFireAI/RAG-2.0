"""
Test script to verify the RAG workflow starts properly for each new conversation.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.rag_2_0.agents.rag_agent import graph

load_dotenv()

def test_conversation_flow():
    """Test that the workflow starts from the beginning for each new conversation."""
    
    print("=== Testing RAG Workflow Conversation Flow ===\n")
    
    # Test 1: First conversation
    print("--- Test 1: First Conversation ---")
    messages1 = [HumanMessage(content="What is machine learning?")]
    
    print(f"Starting with message: {messages1[0].content}")
    result1 = graph.invoke({"messages": messages1})
    
    print(f"Result messages count: {len(result1['messages'])}")
    if result1['messages']:
        print(f"Last message type: {type(result1['messages'][-1])}")
        if hasattr(result1['messages'][-1], 'content'):
            print(f"Last message preview: {result1['messages'][-1].content[:100]}...")
    
    print(f"Final state keys: {list(result1.keys())}")
    print(f"waiting_for_leader: {result1.get('waiting_for_leader', 'Not set')}")
    print(f"detected_leader: {result1.get('detected_leader', 'Not set')}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Second conversation (should start fresh)
    print("--- Test 2: Second Conversation ---")
    messages2 = [HumanMessage(content="How does artificial intelligence work?")]
    
    print(f"Starting with message: {messages2[0].content}")
    result2 = graph.invoke({"messages": messages2})
    
    print(f"Result messages count: {len(result2['messages'])}")
    if result2['messages']:
        print(f"Last message type: {type(result2['messages'][-1])}")
        if hasattr(result2['messages'][-1], 'content'):
            print(f"Last message preview: {result2['messages'][-1].content[:100]}...")
    
    print(f"Final state keys: {list(result2.keys())}")
    print(f"waiting_for_leader: {result2.get('waiting_for_leader', 'Not set')}")
    print(f"detected_leader: {result2.get('detected_leader', 'Not set')}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Conversation with leader specification
    print("--- Test 3: Conversation with Leader Specification ---")
    messages3 = [HumanMessage(content="Janelle, what are your thoughts on business strategy?")]
    
    print(f"Starting with message: {messages3[0].content}")
    result3 = graph.invoke({"messages": messages3})
    
    print(f"Result messages count: {len(result3['messages'])}")
    if result3['messages']:
        print(f"Last message type: {type(result3['messages'][-1])}")
        if hasattr(result3['messages'][-1], 'content'):
            print(f"Last message preview: {result3['messages'][-1].content[:100]}...")
    
    print(f"Final state keys: {list(result3.keys())}")
    print(f"waiting_for_leader: {result3.get('waiting_for_leader', 'Not set')}")
    print(f"detected_leader: {result3.get('detected_leader', 'Not set')}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_conversation_flow() 