import os 
from pathlib import Path

from langchain_google_community import GoogleDriveLoader 

loader = GoogleDriveLoader(
    folder_id="1zLK6qRuQGU1c7Y_d9Th5uQgUF_dss_uH",
    token_path=Path("token.json"),
    num_results=2
)

docs = loader.load()

for d in docs: 
    print(d.page_content[:100]) # first 100 characters