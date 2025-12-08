# Research Paper Summarizer (Multi-PDF RAG)

A full-stack Retrieval-Augmented Generation (RAG) system that ingests multiple research paper PDFs and generates structured summaries, comparative analysis, and question answering with citations.
Built using LangChain, ChromaDB, OpenAI models, and Streamlit.

### ✨ Features
1. Multi-PDF Ingestion

Upload any number of research papers. The system parses each PDF, extracts text, and stores metadata such as filename and page number.

2. Intelligent Chunking

PDFs are chunked using a section-aware text splitter optimized for scientific papers to retain context during retrieval.

3. Vector Database (ChromaDB)

Chunks are embedded using OpenAI embeddings or optional local embeddings.
A persistent vector store enables fast retrieval even across app restarts.

4. RAG-Powered Summaries

Ask questions like:

“Give a structured summary of all uploaded papers.”

“Compare Paper A vs Paper B.”

“Summarize only the methodology sections.”

“What gaps do these papers highlight?”

The model uses retrieved chunks + PDF metadata to generate grounded responses.

5. Citations from Source PDFs

The system displays outputs that reference the original PDFs (via metadata such as source_file).

6. Simple and Fast UI

A clean Streamlit interface for:

Uploading PDFs

Building the RAG index

Entering questions

Viewing generated summaries.

## 🏗️ Architecture Overview
--
┌───────────────────────────────────────────┐
│                 Streamlit UI              │
│  (Upload PDFs, Ask queries, Show results) │
└───────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│           RAG Orchestration Layer         │
│  LangChain pipelines for retrieval + LLM  │
└───────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│           Vector Store (ChromaDB)         │
│   PDF Loading → Chunking → Embeddings     │
└───────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│              LLM Generation               │
│   (GPT-4o-mini / GPT-4.1 / local LLMs)    │
└───────────────────────────────────────────┘
