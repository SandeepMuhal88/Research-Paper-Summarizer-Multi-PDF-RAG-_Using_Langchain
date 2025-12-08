# rag_core/loader.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder_path, fname)
        loader = PyPDFLoader(path)
        loaded = loader.load()
        # Add metadata: paper_id, filename
        for d in loaded:
            d.metadata["source_file"] = fname
        docs.extend(loaded)
    return docs

print("Loader_use secussesfully!!")