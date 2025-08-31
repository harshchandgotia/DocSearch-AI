import os
import sys
import requests
import tempfile
import numpy as np
from pdf2image import convert_from_bytes
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
import easyocr
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from urllib.parse import urlparse

# Local imports
from vector_db import generate_response, retrieval

load_dotenv()


# Load EasyOCR once (faster than per-request)
reader = easyocr.Reader(['en'], gpu=False)

web_app = FastAPI()
API_TOKEN = os.getenv("API_KEY")

# Middleware for API key authentication
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_TOKEN}":
            return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

web_app.add_middleware(AuthMiddleware)

def ocr_page(page_tuple):
    page_num, image = page_tuple
    image_array = np.array(image)
    ocr_results = reader.readtext(image_array)
    texts = [res[1] for res in ocr_results]
    page_texts = "\n".join(texts) if texts else "[No text detected]"
    return page_num, f"-- Page {page_num} --\n{page_texts}"

def extract_text_from_bytes(pdf_bytes):
    # Convert PDF to images
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_bytes(
            pdf_bytes.content,
            dpi=150,
            fmt="jpeg",
            output_folder=temp_dir
        )
        page_image_tuples = list(enumerate(images,start=1))

        all_page_data = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ocr_page, page) for page in page_image_tuples]

            for future in as_completed(futures):
                page_num, page_text = future.result()
                all_page_data.append((page_num, page_text))

        all_page_data.sort(key = lambda x: x[0])
        page_texts = [text for _,text in all_page_data]
        return page_texts

def url_parser(file_url):
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    return file_name 



@web_app.post("/extract", response_class=JSONResponse)
async def text_extraction_pipeline(request: Request):
    data = await request.json()
    urls = data.get("documents")
    if not urls:
        return JSONResponse({"error": "'url' field is required"}, status_code=400)

    try:
        # Fetch PDF
        dictionary = {}
        for url in urls:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            page_texts = extract_text_from_bytes(response)

            # Send to LLM
            pdfData = ""
            pdfData = "<br><br>".join(page_texts)
            pdf_id = str(uuid.uuid4())
            file_name = url_parser(url)
            generate_response(pdfData, pdf_id)
            dictionary[pdf_id] = file_name

        return JSONResponse({"Files uploaded": dictionary}, status_code=200)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    

@web_app.post("/query", response_class=JSONResponse)
async def get_answers(request: Request):
    data = await request.json()
    questions = data.get("questions")

    if not all([questions]):
        return JSONResponse({"error": "'questions' fields are required"}, status_code=400)

    try:
        llm_response = retrieval(questions) 
        return JSONResponse({"answers": llm_response}, status_code=200) 

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

