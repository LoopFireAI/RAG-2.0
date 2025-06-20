from pathlib import Path
from langchain_google_community import GoogleDriveLoader
import hashlib

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"

# Ensure the credentials directory exists
CREDENTIALS_DIR.mkdir(exist_ok=True)

# Set up absolute paths for credentials and token
CREDENTIALS_PATH = CREDENTIALS_DIR / "credentials.json"
TOKEN_PATH = CREDENTIALS_DIR / "token.json"

print(f"Using credentials from: {CREDENTIALS_PATH}")
print(f"Using token from: {TOKEN_PATH}")

# List of folder IDs to process
FOLDER_IDS = [
    "1zLK6qRuQGU1c7Y_d9Th5uQgUF_dss_uH",
    "1yjIqFXi13uO-aGiNPkGPiEkBGGS-3kFZ",
    "1QcaVSSrQm8REMO99cmnivNXrRGuuDELT",
    "1YaaD_hmb4nLgXYWi6aRdV79usTkgiRN8"
]

# Set to store document hashes for deduplication
seen_hashes = set()
all_docs = []
total_docs = 0
duplicate_count = 0

def get_document_hash(doc):
    """Create a hash of the document content and metadata for deduplication."""
    # Combine content and relevant metadata for hashing
    content = doc.page_content
    metadata = doc.metadata
    # Use file name and size as part of the hash if available
    file_info = f"{metadata.get('name', '')}{metadata.get('size', '')}"
    # Create hash
    return hashlib.md5((content + file_info).encode()).hexdigest()

for folder_id in FOLDER_IDS:
    print(f"\nProcessing folder: {folder_id}")

    loader = GoogleDriveLoader(
        folder_id=folder_id,
        token_path=str(TOKEN_PATH),
        credentials_path=str(CREDENTIALS_PATH),
        recursive=True,  # Include subfolders
        load_extended_metadata=True,
        num_results=1  # Get more information about the files
    )

    try:
        docs = loader.load()
        folder_doc_count = len(docs)

        # Process each document and check for duplicates
        for doc in docs:
            doc_hash = get_document_hash(doc)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                all_docs.append(doc)
                total_docs += 1
            else:
                duplicate_count += 1
                print(f"Found duplicate document: {doc.metadata.get('name', 'Unknown')}")

        print(f"Loaded {folder_doc_count} documents from this folder")
        print(f"Unique documents so far: {total_docs}")
        print(f"Duplicates found so far: {duplicate_count}")

        if folder_doc_count > 0:
            print("Sample document metadata:")
            print(docs[0].metadata)
        else:
            print("No documents found in this folder")

    except Exception as e:
        print(f"Error processing folder {folder_id}: {str(e)}")

print(f"\nFinal Summary:")
print(f"Total documents processed: {total_docs + duplicate_count}")
print(f"Unique documents: {total_docs}")
print(f"Duplicates removed: {duplicate_count}")

# Now you can work with all_docs which contains unique documents from all folders
if all_docs:
    print("\nSample of first unique document:")
    print(all_docs[0].page_content[:200] + "...")  # Print first 200 characters

