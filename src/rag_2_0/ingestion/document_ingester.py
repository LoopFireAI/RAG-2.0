"""
Document ingestion script using LangChain integrations.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_community import GoogleDriveLoader

load_dotenv()

class DocumentIngester:
    def __init__(self, data_dir: str = "./data", collection_name: str = "rag_docs"):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            length_function=len,
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def load_documents(self) -> List[Document]:
        """Load documents from Google Drive folder."""
        documents = []
        folder_id = "1zLK6qRuQGU1c7Y_d9Th5uQgUF_dss_uH"
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        token_path = os.getenv("GOOGLE_TOKEN_PATH")

        print(f"Loading documents from Google Drive folder: {folder_id}")
        print(f"Using credentials: {credentials_path}")
        print(f"Using token: {token_path}")

        try:
            loader = GoogleDriveLoader(
                folder_id=folder_id,
                credentials_path=credentials_path,
                token_path=token_path,
                recursive=False,
                num_results=1
            )
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from Google Drive.")
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading documents from Google Drive: {e}")

        # Optionally, print out the first 100 characters of each doc for debugging
        for i, doc in enumerate(documents):
            try:
                print(f"Doc {i+1}: {doc.page_content[:100]}")
            except Exception as e:
                print(f"Error reading content of doc {i+1}: {e}")

        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def ingest_documents(self):
        """Main ingestion process."""
        print("Starting document ingestion...")
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            print("No documents to process.")
            return
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        if not chunks:
            print("No chunks created.")
            return
        
        # Clear existing collection
        try:
            self.vector_store.delete_collection()
            print("Cleared existing collection")
        except:
            pass
        
        # Reinitialize vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Add chunks to vector store
        print(f"Adding {len(chunks)} chunks to vector store...")
        self.vector_store.add_documents(chunks)
        print("Document ingestion completed!")
        
        # Print some stats
        collection = self.vector_store._collection
        print(f"Total documents in collection: {collection.count()}")

def main():
    """Main entry point for document ingestion."""
    ingester = DocumentIngester()
    ingester.ingest_documents()

if __name__ == "__main__":
    main()