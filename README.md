# DocSearch AI

## Overview

DocSearch AI is an intelligent document query system that extracts text from PDFs using OCR and enables semantic search and question-answering through vector databases and LLMs. It supports both direct PDF file uploads and URL-based ingestion, and can handle multiple documents concurrently, each with a unique document ID.

## Features

- **OCR-based Text Extraction**: Uses EasyOCR to process PDF pages, including scanned and image-based documents.
- **Semantic Search & QA**: Employs Pinecone vector store with HuggingFace embeddings for contextual retrieval.
- **LLM-powered Answers**: Integrates Groq (llama-3.3-70b) via a LangChain RetrievalQA chain for precise question answering.
- **Scalable & Fast**: Built on FastAPI with multithreading for concurrent OCR and query processing.
- **Streamlit UI**: A browser-based interface for uploading documents and querying them interactively.

## Architecture

1. **PDF Ingestion**: PDFs arrive either as direct file uploads (`/upload`) or as URLs (`/extract`).
2. **OCR**: Each page is converted to an image via `pdf2image` and processed in parallel with EasyOCR.
3. **Chunking & Embedding**: Extracted text is split into 400-character chunks (100-char overlap) and embedded using `all-MiniLM-L6-v2`.
4. **Indexing**: Embeddings are stored in Pinecone with a `pdf_id` metadata tag for per-document isolation.
5. **Query Handling**: LangChain RetrievalQA retrieves relevant chunks filtered by `pdf_id` and passes them to the Groq LLM for answer generation.

## Installation

```bash
git clone <repo_url>
cd DocSearch-AI
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file (see `.env.example`):

```
API_KEY=your_bearer_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=docsearch-ai
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### 1. Start the API server

```bash
uvicorn main:web_app --reload
```

### 2. Start the Streamlit UI (optional)

```bash
streamlit run app.py
```

### 3. Upload PDF files directly

`POST /upload` — multipart/form-data, field name `files`:

```bash
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "files=@/path/to/document.pdf" \
  -F "files=@/path/to/another.pdf"
```

### 4. Upload PDFs via URL

`POST /extract` — JSON body:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["https://example.com/report.pdf"]}'
```

Both endpoints return the same format:

```json
{
  "Files uploaded": {
    "3f7a8b2c-1234-5678-abcd-ef0123456789": "report.pdf"
  }
}
```

### 5. Query documents

`POST /query` — JSON body. Each item maps a `pdf_id` to a list of questions:

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      {
        "pdf_id": "3f7a8b2c-1234-5678-abcd-ef0123456789",
        "questions": ["What is the summary?", "Who is the author?"]
      }
    ]
  }'
```

Response:

```json
{
  "answers": [
    {
      "pdf_id": "3f7a8b2c-1234-5678-abcd-ef0123456789",
      "answers": [
        "The document summarises...",
        "The author is..."
      ]
    }
  ]
}
```

### 6. Delete a document

`DELETE /documents/{pdf_id}` — removes all embeddings for that document from Pinecone:

```bash
curl -X DELETE http://localhost:8000/documents/3f7a8b2c-1234-5678-abcd-ef0123456789 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Response:

```json
{"deleted": "3f7a8b2c-1234-5678-abcd-ef0123456789"}
```
