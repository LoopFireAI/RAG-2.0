"""
Super simple way to view your Chroma collection
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Initialize with the same embeddings used during ingestion
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Get basic info
collection = vector_store._collection
print(f"Collection name: {collection.name}")
print(f"Total documents: {collection.count()}")

# Get sample documents
# print("\n--- Sample Documents ---")
# results = collection.get()
# for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
#     print(f"\nDocument {i+1}:")
#     print(f"  Content: {doc[:100]}...")
#     print(f"  Metadata: {metadata}")

# Test similarity search using LangChain wrapper
print("\n--- Test Query ---")
query_results = vector_store.similarity_search("can you explain the glass cliff phenomenon?", k=2)
print(f"Query results: {len(query_results)} found")
for i, doc in enumerate(query_results):
    print(f"  Result {i+1}: {doc.page_content[:100]}...")
    print(f"  Result {i+1}: {doc.metadata}")
