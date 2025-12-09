from typing import Dict, Any
import os
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

VECTOR_DIR = "vectorstore"


def load_vectorstore(persist_dir: str = VECTOR_DIR) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectordb


def build_rag_chain(vectordb: Chroma):
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if HF_TOKEN is None:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

    # --- 2. HUGGINGFACE MODEL (IMPORTANT) ---
    endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",   # <-- change model here if needed
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens=500,
        temperature=0.2,
    )

    LLMs = ChatHuggingFace(llm=endpoint)

    system_prompt = """
You are an AI assistant that summarizes and analyzes multiple research papers.

- Use only the provided context.
- When relevant, identify which paper(s) you are using via `source_file` metadata.
- For each answer, clearly distinguish:
  1) Direct facts from the papers
  2) Your synthesized explanation
- If information is unclear or missing in the papers, explicitly say so.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}\n\nContext:\n{context}")
        ]
    )

    def format_docs(docs):
        formatted = []
        for d in docs:
            src = d.metadata.get("source_file", "unknown")
            formatted.append(f"[{src}]\n{d.page_content}")
        return "\n\n-----\n\n".join(formatted)

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough()
        )
        | prompt
        | LLMs
    )

    return rag_chain
