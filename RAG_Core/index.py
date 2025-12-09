from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def chunk_documents( docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200 ) -> List[Document]:
    """This function chunks the documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ", "\n### ", "\n\n", "\n", ". ", "? ", "! ", " ", ""
        ],
    )
    split_docs = splitter.split_documents(docs)
    return split_docs

# print("Yes its working")
