#!/usr/bin/env python3
"""
Comprehensive test script for Wells RAG system
Tests: source retrieval, response generation, and feedback system
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.rag_2_0.agents.rag_agent import create_rag_graph
from langchain_core.messages import HumanMessage

def test_rag_retrieval():
    """Test document retrieval and source inclusion"""
    print("🔍 TESTING RAG DOCUMENT RETRIEVAL")
    print("=" * 50)
    
    rag_graph = create_rag_graph()
    
    # Test queries
    test_queries = [
        "What makes an effective leader?",
        "What are key leadership traits?", 
        "How do leaders build trust?",
        "make me a social media post in Janelle's voice about what makes an effective leader"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 TEST {i}: {query}")
        print("-" * 40)
        
        try:
            result = rag_graph.invoke({
                'messages': [HumanMessage(content=query)]
            })
            
            print(f"✅ Graph executed successfully")
            print(f"📊 Messages generated: {len(result.get('messages', []))}")
            
            # Check each message for sources
            sources_found = False
            main_response = ""
            
            for j, msg in enumerate(result.get('messages', [])):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                if j == 0:
                    print(f"🔸 Input: {content[:100]}...")
                elif len(content) > 100:  # Likely the main response
                    main_response = content
                    sources_found = "Sources:" in content or "📚" in content
                    print(f"🔸 Response length: {len(content)} chars")
                    print(f"🔸 Sources included: {sources_found}")
                    print(f"🔸 Response preview: {content[:200]}...")
            
            if not sources_found and main_response:
                print("⚠️  NO SOURCES FOUND - This is the issue!")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
    print("\n" + "=" * 50)

def test_vector_store():
    """Test if the vector store has documents"""
    print("\n🗄️ TESTING VECTOR STORE")
    print("=" * 50)
    
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        # Test similarity search
        test_query = "leadership traits"
        docs = vectorstore.similarity_search(test_query, k=3)
        
        print(f"📊 Documents found for '{test_query}': {len(docs)}")
        
        if docs:
            for i, doc in enumerate(docs, 1):
                print(f"🔸 Doc {i}: {doc.page_content[:100]}...")
                print(f"   Metadata: {doc.metadata}")
        else:
            print("⚠️  NO DOCUMENTS FOUND IN VECTOR STORE - This explains missing sources!")
            
    except Exception as e:
        print(f"❌ Error accessing vector store: {e}")

def test_slack_bot_functions():
    """Test Slack bot functions individually"""
    print("\n🤖 TESTING SLACK BOT FUNCTIONS")
    print("=" * 50)
    
    try:
        from src.rag_2_0.slack_bot_socket import process_rag_query, clean_response_for_slack
        
        test_query = "What makes an effective leader?"
        print(f"📝 Testing query: {test_query}")
        
        response = process_rag_query(test_query, "test_user", "test_user")
        
        print(f"✅ Response generated: {len(response)} chars")
        print(f"🔍 Sources in response: {'Sources' in response}")
        print(f"📄 Response preview: {response[:200]}...")
        
        # Test response cleaning
        cleaned = clean_response_for_slack(response)
        print(f"🧹 Cleaned response: {len(cleaned)} chars")
        
    except Exception as e:
        print(f"❌ Error testing bot functions: {e}")

if __name__ == "__main__":
    print("🚀 WELLS RAG COMPREHENSIVE TESTING")
    print("=" * 60)
    
    # Test 1: Vector store
    test_vector_store()
    
    # Test 2: RAG retrieval  
    test_rag_retrieval()
    
    # Test 3: Bot functions
    test_slack_bot_functions()
    
    print("\n✅ TESTING COMPLETE")
    print("=" * 60) 