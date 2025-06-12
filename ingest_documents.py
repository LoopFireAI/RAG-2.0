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
        """Load documents from the data directory."""
        documents = []
        
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist.")
            return documents
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                        docs = loader.load()
                        print(f"Loaded PDF: {file_path.name} ({len(docs)} pages)")
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path))
                        docs = loader.load()
                        print(f"Loaded text file: {file_path.name}")
                    else:
                        print(f"Skipping unsupported file: {file_path.name}")
                        continue
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            "source_file": file_path.name,
                            "file_type": file_path.suffix.lower()
                        })
                    
                    documents.extend(docs)
                    
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        
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