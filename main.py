#!/usr/bin/env python3
"""
RAG 2.0 - Main Application Entry Point

A simple command-line interface for the RAG system.
"""

import sys
from typing import Optional
from langchain_core.messages import HumanMessage
from rag_agent import create_rag_graph

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
