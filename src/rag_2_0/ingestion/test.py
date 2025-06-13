import os 
from pathlib import Path

from langchain_google_community import GoogleDriveLoader 

loader = GoogleDriveLoader(
    folder_id="1yjIqFXi13uO-aGiNPkGPiEkBGGS-3kFZ",
    token_path=Path("token.json"),
    num_results=2
)

docs = loader.load()

print(docs)