"""
Document ingestion script using LangChain integrations.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_community import GoogleDriveLoader
import asyncio

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class DocumentIngester:
    def __init__(self, data_dir: str = "./data", collection_name: str = "rag_docs"):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.hash_file = os.getenv("HASH_FILE", "ingested_hashes.txt")

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

    async def load_folder_async(self, folder_id, credentials_path, token_path):
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            credentials_path=credentials_path,
            token_path=token_path,
            recursive=True,
            load_extended_metadata=True,
            load_auth=True
        )
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        return docs

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter metadata to only include simple types and ensure source URL is preserved."""
        # Filter out complex types manually
        filtered_metadata = {}
        for key, value in metadata.items():
            # Only keep simple types that can be serialized
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered_metadata[key] = value
            else:
                # Convert complex types to string representation
                filtered_metadata[key] = str(value)

        # Ensure we keep the source URL if it exists
        if 'source' in metadata:
            filtered_metadata['source'] = metadata['source']

        return filtered_metadata

    def load_documents(self) -> List[Document]:
        """Load documents from Google Drive folders with deduplication, using async parallel loading."""
        documents = []
        seen_hashes = self.load_hashes()
        duplicate_count = 0

        FOLDER_IDS = [
            "1zLK6qRuQGU1c7Y_d9Th5uQgUF_dss_uH",
            "1yjIqFXi13uO-aGiNPkGPiEkBGGS-3kFZ",
            "1QcaVSSrQm8REMO99cmnivNXrRGuuDELT",
            "1YaaD_hmb4nLgXYWi6aRdV79usTkgiRN8",
            "1SBr_c2NwCuXsNZsjau37uxxq2qRQH1F6"
        ]

        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        token_path = os.getenv("GOOGLE_TOKEN_PATH")

        if not credentials_path:
            raise ValueError("GOOGLE_CREDENTIALS_PATH environment variable is not set.")
        if not token_path:
            raise ValueError("GOOGLE_TOKEN_PATH environment variable is not set.")
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at: {credentials_path}.")

        logger.debug(f"GOOGLE_CREDENTIALS_PATH: {credentials_path}")
        logger.debug(f"GOOGLE_TOKEN_PATH: {token_path}")
        logger.debug(f"Credentials file exists: {os.path.exists(credentials_path)}")
        logger.debug(f"Token file exists: {os.path.exists(token_path)}")

        async def gather_all_folders():
            tasks = [
                self.load_folder_async(folder_id, credentials_path, token_path)
                for folder_id in FOLDER_IDS
            ]
            results = await asyncio.gather(*tasks)
            # Flatten the list of lists
            return [doc for folder_docs in results for doc in folder_docs]

        all_docs = asyncio.run(gather_all_folders())

        def get_document_hash(doc):
            content = doc.page_content
            metadata = doc.metadata
            file_info = f"{metadata.get('name', '')}{metadata.get('size', '')}"
            return hashlib.md5((content + file_info).encode()).hexdigest()

        for doc in all_docs:
            doc.metadata = self._filter_metadata(doc.metadata)
            doc_hash = get_document_hash(doc)
            folder_id = doc.metadata.get('parents', ['Unknown'])[0] if doc.metadata.get('parents') else 'Unknown'
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                documents.append(doc)
                logger.info(f"Added unique document: {doc.metadata.get('name', 'Unknown')} from folder: {folder_id}")
                logger.debug("Document Metadata:")
                for key, value in doc.metadata.items():
                    logger.debug(f"  {key}: {value}")
            else:
                duplicate_count += 1
                logger.debug(f"Skipped duplicate document: {doc.metadata.get('name', 'Unknown')} from folder: {folder_id}")

        logger.info("Document Loading Summary:")
        logger.info(f"Total unique documents loaded: {len(documents)}")
        logger.info(f"Total duplicates skipped: {duplicate_count}")

        self.save_hashes(seen_hashes)
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

        # Filter metadata for each chunk
        for chunk in chunks:
            chunk.metadata = self._filter_metadata(chunk.metadata)

        # Log metadata for first chunk of each document
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:  # Log every 10th chunk to avoid too much output
                logger.debug(f"Chunk {i+1} Metadata:")
                for key, value in chunk.metadata.items():
                    logger.debug(f"  {key}: {value}")

        return chunks

    def ingest_documents(self):
        """Main ingestion process."""
        logger.info("Starting document ingestion...")

        try:
            # Load documents
            documents = self.load_documents()
            if not documents:
                logger.warning("No documents to process.")
                return

            # Chunk documents
            chunks = self.chunk_documents(documents)
            if not chunks:
                logger.warning("No chunks created.")
                return

            # Clear existing collection
            # try:
            #     # self.vector_store.delete_collection()
            #     # print("Cleared existing collection")
            # except:
            #     pass

            # Reinitialize vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )

            # Add chunks to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            self.vector_store.add_documents(chunks)
            logger.info("Document ingestion completed!")

            # Print some stats
            collection = self.vector_store._collection
            logger.info(f"Total documents in collection: {collection.count()}")
        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise

    def load_hashes(self):
        if not os.path.exists(self.hash_file):
            return set()
        with open(self.hash_file, "r") as f:
            return set(line.strip() for line in f)

    def save_hashes(self, hashes):
        with open(self.hash_file, "w") as f:
            for h in hashes:
                f.write(h + "\n")

def main():
    """Main entry point for document ingestion."""
    ingester = DocumentIngester()
    ingester.ingest_documents()

if __name__ == "__main__":
    main()
