import asyncio
import os
import tempfile
import uuid
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import partial
from urllib.parse import urlparse

import numpy as np
import requests
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
from dotenv import load_dotenv
import easyocr
from pdf2image import convert_from_bytes

from config import load_config
from vector_db import ingest_document, run_query, delete_document
from database import (
    create_session,
    get_session,
    list_sessions,
    delete_session,
    add_message,
    get_messages,
    update_session_title,
    update_session_pdf_ids,
    add_document,
    list_documents,
    delete_document_record,
)

load_dotenv()
_config = load_config()
_ocr_max_workers = _config.get("processing", {}).get("ocr_max_workers", 4)


# --- Pydantic models ---

class ExtractRequest(BaseModel):
    documents: List[str]

    @field_validator("documents")
    @classmethod
    def documents_not_empty(cls, v):
        if not v:
            raise ValueError("documents must be a non-empty list")
        return v


class SingleQueryRequest(BaseModel):
    question: str
    pinned_pdf_ids: List[str]
    session_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    pinned_pdf_ids: List[str]
    title: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    pinned_pdf_ids: Optional[List[str]] = None
    title: Optional[str] = None


# Load EasyOCR lazily to avoid blocking server startup
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


web_app = FastAPI()
API_TOKEN = os.getenv("API_KEY")


# --- Auth middleware ---

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_TOKEN}":
            return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

web_app.add_middleware(AuthMiddleware)


# --- OCR helpers ---

def _ocr_page(page_tuple):
    page_num, image = page_tuple
    image_array = np.array(image)
    ocr_results = _get_reader().readtext(image_array)
    texts = [res[1] for res in ocr_results]
    page_text = "\n".join(texts) if texts else "[No text detected]"
    return page_num, f"-- Page {page_num} --\n{page_text}"


def _extract_text_from_bytes(pdf_bytes: bytes) -> list[str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=150,
            fmt="jpeg",
            output_folder=temp_dir,
        )
        page_image_tuples = list(enumerate(images, start=1))

        all_page_data = []
        with ThreadPoolExecutor(max_workers=_ocr_max_workers) as executor:
            futures = [executor.submit(_ocr_page, page) for page in page_image_tuples]
            for future in as_completed(futures):
                all_page_data.append(future.result())

        all_page_data.sort(key=lambda x: x[0])
        return [text for _, text in all_page_data]


def _url_parser(file_url: str) -> str:
    return os.path.basename(urlparse(file_url).path)


def _process_pdf_bytes(raw_bytes: bytes, filename: str) -> tuple:
    """Shared ingestion pipeline used by both /extract and /upload."""
    page_texts = _extract_text_from_bytes(raw_bytes)
    pdf_data = "\n\n".join(page_texts)
    pdf_id = str(uuid.uuid4())
    ingest_document(pdf_data, pdf_id)
    return pdf_id, filename


# --- Endpoints ---

@web_app.post("/extract", response_class=JSONResponse)
async def text_extraction_pipeline(body: ExtractRequest):
    try:
        async def _process_url(url: str):
            response = await asyncio.to_thread(
                partial(requests.get, url, timeout=30)
            )
            response.raise_for_status()
            return await asyncio.to_thread(_process_pdf_bytes, response.content, _url_parser(url))

        tasks = [_process_url(url) for url in body.documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        dictionary = {}
        errors = []
        for url, result in zip(body.documents, results):
            if isinstance(result, Exception):
                errors.append(f"{url}: {result}")
            else:
                pdf_id, filename = result
                dictionary[pdf_id] = filename
                add_document(pdf_id, filename)

        response_body: dict = {"Files uploaded": dictionary}
        if errors:
            response_body["errors"] = errors
        return JSONResponse(response_body, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.post("/upload", response_class=JSONResponse)
async def file_upload_pipeline(files: List[UploadFile] = File(...)):
    if not files:
        return JSONResponse({"error": "No files provided"}, status_code=400)
    try:
        dictionary = {}
        for file in files:
            raw_bytes = await file.read()
            pdf_id, filename = await asyncio.to_thread(
                _process_pdf_bytes, raw_bytes, file.filename or "unknown.pdf"
            )
            dictionary[pdf_id] = filename
            add_document(pdf_id, filename)
        return JSONResponse({"Files uploaded": dictionary}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.post("/query", response_class=JSONResponse)
async def query_documents(body: SingleQueryRequest):
    try:
        conversation_history = []
        session_id = body.session_id
        pinned_pdf_ids = body.pinned_pdf_ids

        if session_id:
            session = get_session(session_id)
            if not session:
                return JSONResponse({"error": "Session not found"}, status_code=404)
            messages = get_messages(session_id)
            conversation_history = [
                {"role": m["role"], "content": m["content"]} for m in messages
            ]
            # Fall back to the session's pinned_pdf_ids when the request list is empty
            if not pinned_pdf_ids:
                pinned_pdf_ids = session.get("pinned_pdf_ids", [])

        result = await asyncio.to_thread(
            run_query, body.question, pinned_pdf_ids, conversation_history
        )

        if session_id:
            add_message(session_id, "user", body.question)
            add_message(session_id, "assistant", result["answer"], metadata={
                "is_supported": result["is_supported"],
                "is_useful": result["is_useful"],
                "revision_count": result["revision_count"],
                "rewrite_count": result["rewrite_count"],
                "sources": result["sources"],
                "retrieval_used": result["retrieval_used"],
            })

            # Auto-generate session title from first question
            if not get_session(session_id).get("title"):
                update_session_title(session_id, body.question[:60])

        return JSONResponse({
            "answer": result["answer"],
            "is_supported": result["is_supported"],
            "is_useful": result["is_useful"],
            "revision_count": result["revision_count"],
            "rewrite_count": result["rewrite_count"],
            "sources": result["sources"],
            "retrieval_used": result["retrieval_used"],
            "no_answer": result["no_answer"],
            "session_id": session_id,
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.post("/sessions", response_class=JSONResponse)
async def create_new_session(body: CreateSessionRequest):
    try:
        title = body.title or ""
        session_id = create_session(body.pinned_pdf_ids, title)
        session = get_session(session_id)
        return JSONResponse({
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "pinned_pdf_ids": session["pinned_pdf_ids"],
            "title": session["title"],
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.get("/sessions", response_class=JSONResponse)
async def get_all_sessions():
    try:
        sessions = list_sessions()
        return JSONResponse({"sessions": sessions}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.get("/sessions/{session_id}/messages", response_class=JSONResponse)
async def get_session_messages(session_id: str):
    try:
        messages = get_messages(session_id)
        return JSONResponse({
            "session_id": session_id,
            "messages": messages,
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.delete("/sessions/{session_id}", response_class=JSONResponse)
async def remove_session(session_id: str):
    try:
        delete_session(session_id)
        return JSONResponse({"deleted": session_id}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.patch("/sessions/{session_id}", response_class=JSONResponse)
async def update_existing_session(session_id: str, body: UpdateSessionRequest):
    try:
        session = get_session(session_id)
        if not session:
            return JSONResponse({"error": "Session not found"}, status_code=404)

        if body.pinned_pdf_ids is not None:
            update_session_pdf_ids(session_id, body.pinned_pdf_ids)
        if body.title is not None:
            update_session_title(session_id, body.title)

        updated_session = get_session(session_id)
        return JSONResponse({
            "session_id": updated_session["session_id"],
            "pinned_pdf_ids": updated_session["pinned_pdf_ids"],
            "title": updated_session["title"],
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.get("/documents", response_class=JSONResponse)
async def get_all_documents():
    try:
        docs = list_documents()
        return JSONResponse({"documents": docs}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.delete("/documents/{pdf_id}", response_class=JSONResponse)
async def remove_document(pdf_id: str):
    try:
        delete_document(pdf_id)
        delete_document_record(pdf_id)
        return JSONResponse({"deleted": pdf_id}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
