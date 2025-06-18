"""
Super simple way to view your Chroma collection
"""

import chromadb
from chromadb.config import Settings

# Connect to your existing collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("rag_docs")

# Get basic info
print(f"Collection name: {collection.name}")
print(f"Total documents: {collection.count()}")

# Get a few sample documents
print("\n--- Sample Documents ---")
results = collection.get(limit=5)
for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
    print(f"\nDocument {i+1}:")
    print(f"  Content: {doc[:100]}...")
    print(f"  Metadata: {metadata}")

# You can also query by text
print("\n--- Test Query ---")
query_results = collection.query(
    query_texts=["What is this about?"],
    n_results=2
)
print(f"Query results: {len(query_results['documents'][0])} found")
for i, doc in enumerate(query_results['documents'][0]):
    print(f"  Result {i+1}: {doc[:100]}...") 