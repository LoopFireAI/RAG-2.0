#!/usr/bin/env python3
"""
RAG 2.0 - Main Application Entry Point

A simple command-line interface for the RAG system.
"""

import sys
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_2_0.agents.rag_agent import create_rag_graph

def run_rag_query(query: str) -> None:
    """Run a single RAG query and display results."""
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        graph = create_rag_graph()
        result = graph.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        print(f"Documents retrieved: {len(result.get('documents', []))}")
        print(f"Response: {result['messages'][-1].content}")
        
        # Test feedback collection (simulate user feedback)
        try:
            from rag_2_0.feedback.feedback_collector import FeedbackCollector
            from rag_2_0.feedback.feedback_storage import FeedbackStorage
            
            storage = FeedbackStorage()
            collector = FeedbackCollector(storage)
            
            response_id = result.get('response_id')
            if response_id:
                print("\n" + "="*50)
                print("📊 Testing Feedback System")
                print("="*50)
                print("Simulating user feedback: Rating = 4, Relevance = 3")
                
                # Check if response is in cache
                print(f"🔍 Checking if response {response_id} is in cache...")
                print(f"🔍 Cache keys: {list(collector.response_cache.keys())}")
                
                # Simulate feedback collection
                feedback = collector.collect_feedback_simple(
                    response_id=response_id,
                    satisfaction=4,
                    relevance=3,
                    text="Good response, helpful examples"
                )
                
                if feedback:
                    print("✅ Feedback collected successfully!")
                    print(f"📝 Stored feedback for query: '{feedback['query'][:50]}...'")
                else:
                    print("❌ Feedback collection failed")
                    
        except ImportError:
            print("\n❌ Feedback system not available (import error)")
        except Exception as e:
            print(f"\n❌ Feedback collection error: {e}")
        
    except Exception as e:
        print(f"Error processing query: {e}")
        sys.exit(1)

def interactive_mode() -> None:
    """Run in interactive mode for continuous queries."""
    print("RAG 2.0 - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50)
    
    graph = create_rag_graph()
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                continue
                
            print(f"\nProcessing: {query}")
            print("-" * 30)
            
            result = graph.invoke({
                "messages": [HumanMessage(content=query)]
            })
            
            print(f"Documents found: {len(result.get('documents', []))}")
            print(f"Answer: {result['messages'][-1].content}")
            
            # Collect feedback if enabled
            try:
                from rag_2_0.feedback.feedback_collector import FeedbackCollector
                from rag_2_0.feedback.feedback_storage import FeedbackStorage
                
                storage = FeedbackStorage()
                collector = FeedbackCollector(storage)
                
                response_id = result.get('response_id')
                if response_id and collector.should_prompt_feedback(response_id):
                    collector.collect_feedback_interactive(response_id)
                    
            except ImportError:
                pass  # Feedback system not available
            except Exception as e:
                print(f"[DEBUG] Feedback collection error: {e}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main application entry point."""
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        run_rag_query(query)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
