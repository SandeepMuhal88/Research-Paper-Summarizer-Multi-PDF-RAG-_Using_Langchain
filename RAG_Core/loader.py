# rag_core/loader.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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

# print("Loader_use secussesfully!!")
def build_vectorstore(docs: List[Document],persist_dir: str = "vectorstore/chroma",use_local_embeddings: bool = False) -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectordb.persist()
    return vectordb

def load_vectorstore(
    persist_dir: str = "vectorstore/chroma",use_local_embeddings: bool = False) -> Chroma:
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vectordb
    