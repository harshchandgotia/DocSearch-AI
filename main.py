import os
import requests
import tempfile
import numpy as np
from pdf2image import convert_from_bytes
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, field_validator
from typing import List
from dotenv import load_dotenv
import easyocr
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from urllib.parse import urlparse

from vector_db import ingest_document, retrieval, delete_document

load_dotenv()


# --- Pydantic models ---

class ExtractRequest(BaseModel):
    documents: List[str]

    @field_validator("documents")
    @classmethod
    def documents_not_empty(cls, v):
        if not v:
            raise ValueError("documents must be a non-empty list")
        return v


class QuestionItem(BaseModel):
    pdf_id: str
    questions: List[str]

    @field_validator("questions")
    @classmethod
    def questions_not_empty(cls, v):
        if not v:
            raise ValueError("questions must be a non-empty list")
        return v


class QueryRequest(BaseModel):
    questions: List[QuestionItem]

    @field_validator("questions")
    @classmethod
    def questions_list_not_empty(cls, v):
        if not v:
            raise ValueError("questions list must be non-empty")
        return v


# Load EasyOCR once (expensive to initialise per-request)
reader = easyocr.Reader(['en'], gpu=False)

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

def ocr_page(page_tuple):
    page_num, image = page_tuple
    image_array = np.array(image)
    ocr_results = reader.readtext(image_array)
    texts = [res[1] for res in ocr_results]
    page_text = "\n".join(texts) if texts else "[No text detected]"
    return page_num, f"-- Page {page_num} --\n{page_text}"


def extract_text_from_bytes(pdf_bytes: bytes):
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=150,
            fmt="jpeg",
            output_folder=temp_dir
        )
        page_image_tuples = list(enumerate(images, start=1))

        all_page_data = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ocr_page, page) for page in page_image_tuples]
            for future in as_completed(futures):
                page_num, page_text = future.result()
                all_page_data.append((page_num, page_text))

        all_page_data.sort(key=lambda x: x[0])
        return [text for _, text in all_page_data]


def url_parser(file_url):
    parsed_url = urlparse(file_url)
    return os.path.basename(parsed_url.path)


def _process_pdf_bytes(raw_bytes: bytes, filename: str) -> tuple:
    """Shared ingestion pipeline used by both /extract and /upload."""
    page_texts = extract_text_from_bytes(raw_bytes)
    pdf_data = "<br><br>".join(page_texts)
    pdf_id = str(uuid.uuid4())
    ingest_document(pdf_data, pdf_id)
    return pdf_id, filename


# --- Endpoints ---

@web_app.post("/extract", response_class=JSONResponse)
async def text_extraction_pipeline(body: ExtractRequest):
    try:
        dictionary = {}
        for url in body.documents:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            pdf_id, filename = _process_pdf_bytes(response.content, url_parser(url))
            dictionary[pdf_id] = filename
        return JSONResponse({"Files uploaded": dictionary}, status_code=200)
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
            pdf_id, filename = _process_pdf_bytes(raw_bytes, file.filename or "unknown.pdf")
            dictionary[pdf_id] = filename
        return JSONResponse({"Files uploaded": dictionary}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.post("/query", response_class=JSONResponse)
async def get_answers(body: QueryRequest):
    try:
        questions_payload = [item.model_dump() for item in body.questions]
        llm_response = retrieval(questions_payload)
        return JSONResponse({"answers": llm_response}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@web_app.delete("/documents/{pdf_id}", response_class=JSONResponse)
async def remove_document(pdf_id: str):
    try:
        delete_document(pdf_id)
        return JSONResponse({"deleted": pdf_id}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
