import os
import fitz  # PyMuPDF
import io
import base64
import zipfile
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Groq API Key (Hardcoded for local testing — DO NOT share publicly)
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

# Directories
INPUT_DIR = "./inputs"
RAW_DIR = "./outputs"
STRUCTURED_DIR = "./structured_outputs"
SCHEMA_PATH = "./invoice_target_schema.json"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(STRUCTURED_DIR, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def pdf_to_base64_images(pdf_path):
    doc = fitz.open(pdf_path)
    base64_images = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        temp_file = f"temp_page_{i}.png"
        img.save(temp_file, format="PNG")
        base64_images.append(encode_image(temp_file))
        os.remove(temp_file)
    return base64_images

def extract_with_groq(base64_image):
    system_prompt = """
    Extract all data from this hotel invoice as structured JSON.
    Group by logical sections like: Guest Info, Charges, Taxes, Totals, etc.
    Include blank fields as null. Do not guess. If empty, return {}.
    """
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract invoice data as structured JSON"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]}
            ],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Groq Error: {str(e)}"}

def transform_to_schema(raw_json, schema):
    prompt = f"""
    Reformat OCR invoice data into this fixed schema:
    - Translate all values to English.
    - Use null for missing fields.
    - Format dates as YYYY-MM-DD.
    - Strictly follow this schema:
    {json.dumps(schema, indent=2)}
    """
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Raw JSON:\n{json.dumps(raw_json)}"}
                ]}
            ],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Transform Error: {str(e)}"}

@app.post("/upload-invoices/")
async def upload_invoices(files: List[UploadFile] = File(...)):
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    structured_paths = []

    for file in files:
        filename = file.filename
        input_path = os.path.join(INPUT_DIR, filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        base64_images = pdf_to_base64_images(input_path)
        all_raw_data = []
        for base64_image in base64_images:
            raw = extract_with_groq(base64_image)
            all_raw_data.append(raw)

        merged_raw = {k: v for d in all_raw_data if isinstance(d, dict) for k, v in d.items()}

        # Save raw output
        raw_path = os.path.join(RAW_DIR, filename.replace(".pdf", "_raw.json"))
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(merged_raw, f, indent=2, ensure_ascii=False)

        # Save structured output
        structured = transform_to_schema(merged_raw, schema)
        structured_path = os.path.join(STRUCTURED_DIR, filename.replace(".pdf", "_structured.json"))
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)

        structured_paths.append(structured_path)

    # Create ZIP of structured outputs
    zip_path = os.path.join(STRUCTURED_DIR, "structured_invoices.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for path in structured_paths:
            zipf.write(path, os.path.basename(path))

    return FileResponse(zip_path, media_type="application/zip", filename="structured_invoices.zip")
