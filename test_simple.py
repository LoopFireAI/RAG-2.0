#!/usr/bin/env python3
"""Quick test of the simple RAG agent."""

from rag_agent import create_rag_graph
from langchain_core.messages import HumanMessage

def test_rag():
    """Test the RAG agent."""
    graph = create_rag_graph()
    
    # Test input
    result = graph.invoke({
        "messages": [HumanMessage(content="What is machine learning?")]
    })
    
    print("Query:", result["query"])
    print("Documents found:", len(result["documents"]))
    print("Response:", result["messages"][-1].content)

if __name__ == "__main__":
    test_rag()