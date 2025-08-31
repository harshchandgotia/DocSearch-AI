# DocSearch–AI 

## Overview

DocSearch is an intelligent document query system that extracts text from PDFs using OCR and enables semantic search and question-answering through vector databases and LLMs. It is designed for fast, accurate, and concurrent querying of large documents. It can ingest multiple documents in the vector database, each with a unique document ID of their own.

## Features

* **OCR-based Text Extraction**: Uses EasyOCR to process PDF pages.
* **Semantic Search & QA**: Employs Pinecone vector store with HuggingFace embeddings for contextual retrieval.
* **LLM-powered Answers**: Integrates Groq/OpenAI models for precise question answering.
* **Scalable & Fast**: Built on FastAPI with multithreading for concurrent requests.

## Architecture

1. **PDF Upload & Conversion**: Converts PDFs into images using `pdf2image`.
2. **Text Extraction**: Extracts text from images via EasyOCR.
3. **Vectorization**: Splits extracted text and embeds it using `all-MiniLM-L6-v2`.
4. **Indexing**: Stores embeddings in Pinecone for semantic retrieval, with metadata linking them to their document ID.
5. **Query Handling**: Uses LangChain RetrievalQA to fetch and answer user questions, which are mapped to document IDs.

## Installation

```bash
git clone <repo_url>
cd DocSearch
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:

```
API_KEY=authorization_key
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

## Usage

### 1. Start the Server

```bash
uvicorn main:web_app --reload
```

### 2. Extract Text from PDF

POST request to `/extract` with JSON:

```json
{
  "documents": ["https://example1.com/sample.pdf", "https://example2.com/sample.pdf"]
}
```

### 3. Query Extracted Data

POST request to `/query` with JSON:

```json
{
  "questions": {"doc_id": "What is the summary?", "doc_id": "Who is the author?"}
}
```

### 4. Receive Answers

Response:

```json
{
  "answers": ["Summary: ...", "Author: ..."]
}
```


