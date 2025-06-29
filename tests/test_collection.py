"""
Simple script to test the rag_docs collection directly.
Run this and then use the interactive functions below.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Initialize the same vector store as in your DocumentIngester
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Get the collection directly
collection = vector_store._collection

def check_collection():
    """Check basic collection info"""
    total_docs = collection.count()
    print(f"Total documents in rag_docs collection: {total_docs}")
    return total_docs

def get_sample_docs(limit=3):
    """Get a few sample documents"""
    sample = collection.get(limit=limit)
    print(f"\nRetrieved {len(sample['documents'])} sample documents:")
    for i, (doc, metadata) in enumerate(zip(sample['documents'], sample['metadatas'])):
        print(f"\nDocument {i+1}:")
        print(f"  Content preview: {doc[:150]}...")
        print(f"  Metadata: {metadata}")

def search_docs(query, k=3):
    """Search for documents using similarity search"""
    print(f"\nSearching for: '{query}'")
    results = vector_store.similarity_search(query, k=k)
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Content: {doc.page_content[:150]}...")
        print(f"  Metadata: {doc.metadata}")

def list_metadata_fields():
    """List all available metadata fields"""
    all_docs = collection.get()
    if all_docs['metadatas']:
        all_keys = set()
        for metadata in all_docs['metadatas']:
            if metadata:
                all_keys.update(metadata.keys())
        print(f"Available metadata fields: {sorted(all_keys)}")
        return sorted(all_keys)
    return []

# Run basic check
print("=== Testing rag_docs Collection ===")
total = check_collection()

if total > 0:
    print("\n✅ Collection has documents!")
    
    # Get sample documents
    get_sample_docs(2)
    
    # List metadata fields
    print("\n--- Metadata Fields ---")
    fields = list_metadata_fields()
    
    # Test a search
    print("\n--- Test Search ---")
    search_docs("What is this about?", k=2)
    
    print("\n=== Collection Test Complete ===")
    print("\nYou can now use these functions interactively:")
    print("- check_collection()")
    print("- get_sample_docs(limit=5)")
    print("- search_docs('your query here', k=3)")
    print("- list_metadata_fields()")
else:
    print("❌ No documents found in collection!") 