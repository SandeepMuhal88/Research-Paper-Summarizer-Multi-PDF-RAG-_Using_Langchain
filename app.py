# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from RAG_Core.loader import load_pdfs_from_folder, build_vectorstore,load_vectorstore
from RAG_Core.rag_chain import build_rag_chain
from RAG_Core.index import chunk_documents

load_dotenv()
DATA_FOLDER = "data/raw_pdfs"
VECTOR_DIR = "vectorstore"
os.makedirs(DATA_FOLDER, exist_ok=True)
st.set_page_config(page_title="Research Paper Summarizer (Multi-PDF RAG)")
st.title("📚 Research Paper Summarizer (Multi-PDF RAG)")

# 1. File upload
uploaded_files = st.file_uploader(
    "Upload research paper PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Saving uploaded PDFs...")
    for f in uploaded_files:
        path = os.path.join(DATA_FOLDER, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
    st.success("Files saved. You can now (re)build the index.")

# 2. Build/refresh index
if st.button("Build / Refresh Index"):
    with st.spinner("Loading and indexing PDFs..."):
        docs = load_pdfs_from_folder(DATA_FOLDER)
        chunks = chunk_documents(docs)
        _ = build_vectorstore(chunks, persist_dir=VECTOR_DIR)
    st.success("Index built successfully!")

# 3. Load vectorstore & build chain (lazy)
@st.cache_resource
def get_rag_chain():
    vectordb = load_vectorstore()
    return build_rag_chain(vectordb)

st.subheader("Ask Questions / Get Summaries")

query = st.text_area(
    "Enter your question or summarization request",
    value="Give me a structured summary of all uploaded papers."
)

if st.button("Run RAG") and query.strip():
    chain = get_rag_chain()
    with st.spinner("Thinking..."):
        result = chain.invoke(query)
    st.markdown("### Answer")
    st.write(result.content)