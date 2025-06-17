"""
Document ingestion script using LangChain integrations.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_community import GoogleDriveLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

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
    
    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter metadata to only include simple types and ensure source URL is preserved."""
        # First filter out complex types
        filtered_metadata = filter_complex_metadata(metadata)
        
        # Ensure we keep the source URL if it exists
        if 'source' in metadata:
            filtered_metadata['source'] = metadata['source']
        
        return filtered_metadata
    
    def load_documents(self) -> List[Document]:
        """Load documents from Google Drive folders with deduplication."""
        documents = []
        seen_hashes = set()
        duplicate_count = 0
        
        # List of folder IDs to process
        FOLDER_IDS = [
            "1zLK6qRuQGU1c7Y_d9Th5uQgUF_dss_uH", 
            "1yjIqFXi13uO-aGiNPkGPiEkBGGS-3kFZ", 
            "1QcaVSSrQm8REMO99cmnivNXrRGuuDELT", 
            "1YaaD_hmb4nLgXYWi6aRdV79usTkgiRN8"
        ]
        
        # Get credentials and token paths from environment variables
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        token_path = os.getenv("GOOGLE_TOKEN_PATH")

        # Validate required environment variables
        if not credentials_path:
            raise ValueError(
                "GOOGLE_CREDENTIALS_PATH environment variable is not set. "
                "Please set it to the path of your Google credentials JSON file."
            )
        
        if not token_path:
            raise ValueError(
                "GOOGLE_TOKEN_PATH environment variable is not set. "
                "Please set it to the path where you want to store the token file."
            )

        # Validate that the files exist
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found at: {credentials_path}. "
                "Please ensure the path is correct and the file exists."
            )

        def get_document_hash(doc):
            """Create a hash of the document content and metadata for deduplication."""
            content = doc.page_content
            metadata = doc.metadata
            file_info = f"{metadata.get('name', '')}{metadata.get('size', '')}"
            return hashlib.md5((content + file_info).encode()).hexdigest()

        for folder_id in FOLDER_IDS:
            print(f"\nProcessing folder: {folder_id}")
            print(f"Using credentials: {credentials_path}")
            print(f"Using token: {token_path}")

            try:
                loader = GoogleDriveLoader(
                    folder_id=folder_id,
                    credentials_path=credentials_path,
                    token_path=token_path,
                    recursive=True,  # Include subfolders
                    load_extended_metadata=True,
                    load_auth=True
                )
                docs = loader.load()
                print(f"Loaded {len(docs)} documents from folder {folder_id}")
                
                # Process each document and check for duplicates
                for doc in docs:
                    doc.metadata = self._filter_metadata(doc.metadata)
                    doc_hash = get_document_hash(doc)
                    
                    if doc_hash not in seen_hashes:
                        seen_hashes.add(doc_hash)
                        documents.append(doc)
                        print(f"\nAdded unique document: {doc.metadata.get('name', 'Unknown')}")
                        print("Document Metadata:")
                        for key, value in doc.metadata.items():
                            print(f"  {key}: {value}")
                    else:
                        duplicate_count += 1
                        print(f"Skipped duplicate document: {doc.metadata.get('name', 'Unknown')}")
                
            except Exception as e:
                print(f"Error loading documents from folder {folder_id}: {e}")
                continue  # Continue with next folder even if one fails

        print(f"\nDocument Loading Summary:")
        print(f"Total unique documents loaded: {len(documents)}")
        print(f"Total duplicates skipped: {duplicate_count}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Filter metadata for each chunk
        for chunk in chunks:
            chunk.metadata = self._filter_metadata(chunk.metadata)
        
        # Log metadata for first chunk of each document
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:  # Log every 10th chunk to avoid too much output
                print(f"\nChunk {i+1} Metadata:")
                for key, value in chunk.metadata.items():
                    print(f"  {key}: {value}")
        
        return chunks
    
    def ingest_documents(self):
        """Main ingestion process."""
        print("Starting document ingestion...")
        
        try:
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
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            raise

def main():
    """Main entry point for document ingestion."""
    ingester = DocumentIngester()
    ingester.ingest_documents()

if __name__ == "__main__":
    main()