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

# Local imports
from vector_db import generate_response

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

@web_app.post("/", response_class=JSONResponse)
async def text_extraction_pipeline(request: Request):
    data = await request.json()
    url = data.get("documents")
    questions = data.get("questions")
    if not url:
        return JSONResponse({"error": "'url' field is required"}, status_code=400)

    try:
        # Fetch PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Convert PDF to images
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_bytes(
                response.content,
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

        # Send to LLM
        pdfData = "<br><br>".join(page_texts)
        llm_response = generate_response(pdfData, questions)

        return JSONResponse({"answers": llm_response}, status_code=200)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


