from typing import Dict, Any
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

def build_rag_chain(vectordb: Chroma):
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )   
    LLMS=ChatHuggingFace(llm=llm)
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
        | LLMS
    )

    return rag_chain

print("rag_chain_use_successfully!!")