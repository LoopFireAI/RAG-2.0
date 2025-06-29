"""
Test script to verify documents are properly stored in the vector database.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

def test_vector_store():
    """Test if documents are properly stored and can be retrieved."""
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vector_store = Chroma(
        collection_name="rag_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Get collection info
    collection = vector_store._collection
    total_docs = collection.count()
    print(f"Total documents in collection: {total_docs}")
    
    if total_docs == 0:
        print("❌ No documents found in the vector store!")
        return
    
    print("✅ Documents found in vector store!")
    
    # Test 1: Get a sample of documents
    print("\n--- Test 1: Sample Documents ---")
    try:
        # Get a few documents to see their content
        sample_docs = collection.get(limit=3)
        print(f"Retrieved {len(sample_docs['documents'])} sample documents")
        
        for i, (doc, metadata) in enumerate(zip(sample_docs['documents'], sample_docs['metadatas'])):
            print(f"\nDocument {i+1}:")
            print(f"  Content preview: {doc[:100]}...")
            print(f"  Metadata keys: {list(metadata.keys())}")
            if 'name' in metadata:
                print(f"  Document name: {metadata['name']}")
            if 'source' in metadata:
                print(f"  Source: {metadata['source']}")
    except Exception as e:
        print(f"❌ Error retrieving sample documents: {e}")
    
    # Test 2: Test similarity search
    print("\n--- Test 2: Similarity Search ---")
    test_queries = [
        "What is the main topic?",
        "How does this work?",
        "What are the key points?",
        "Can you explain this concept?"
    ]
    
    for query in test_queries:
        try:
            print(f"\nSearching for: '{query}'")
            results = vector_store.similarity_search(query, k=2)
            print(f"  Found {len(results)} results")
            
            for j, doc in enumerate(results):
                print(f"  Result {j+1}:")
                print(f"    Content preview: {doc.page_content[:100]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    if 'name' in doc.metadata:
                        print(f"    Document: {doc.metadata['name']}")
                    if 'source' in doc.metadata:
                        print(f"    Source: {doc.metadata['source']}")
        except Exception as e:
            print(f"  ❌ Error searching for '{query}': {e}")
    
    # Test 3: Test metadata filtering
    print("\n--- Test 3: Metadata Analysis ---")
    try:
        # Get all metadata to see what fields are available
        all_docs = collection.get()
        if all_docs['metadatas']:
            # Find all unique metadata keys
            all_keys = set()
            for metadata in all_docs['metadatas']:
                if metadata:
                    all_keys.update(metadata.keys())
            
            print(f"Available metadata fields: {sorted(all_keys)}")
            
            # Count documents by some common metadata fields
            if 'name' in all_keys:
                names = [m.get('name', 'Unknown') for m in all_docs['metadatas'] if m]
                unique_names = set(names)
                print(f"Unique document names: {len(unique_names)}")
                print(f"Sample names: {list(unique_names)[:5]}")
            
            if 'source' in all_keys:
                sources = [m.get('source', 'Unknown') for m in all_docs['metadatas'] if m]
                unique_sources = set(sources)
                print(f"Unique sources: {len(unique_sources)}")
                print(f"Sample sources: {list(unique_sources)[:3]}")
                
    except Exception as e:
        print(f"❌ Error analyzing metadata: {e}")
    
    print("\n--- Vector Store Test Complete ---")

if __name__ == "__main__":
    test_vector_store() 