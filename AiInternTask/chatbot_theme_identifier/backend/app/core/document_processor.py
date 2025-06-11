import os
import uuid
import requests
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
from ..config import settings, _embedder, _pc
from docx import Document

def extract_text_from_txt(txt_path):
    with open(txt_path, encoding='utf-8') as f:
        return f.read()
    
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = []
    current_page_text = []
    page_number = 1

    for para in doc.paragraphs:
        # Check if paragraph contains a manual page break in any run
        has_page_break = any('pageBreak' in run._element.xml for run in para.runs)
        if has_page_break and current_page_text:
            # Save current page before starting a new one
            text.append({"page": page_number, "text": "\n".join(current_page_text)})
            page_number += 1
            current_page_text = []

        # Add paragraph text if not empty
        if para.text.strip():
            current_page_text.append(para.text.strip())

    # Add the last page (or whole document if no breaks)
    if current_page_text:
        text.append({"page": page_number, "text": "\n".join(current_page_text)})

    return text

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text.append({"page": i + 1, "text": page_text})
    return text

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def process_and_split_document(file_path, doc_name, doc_id):
    """
    Handles PDF, images, .txt, .docx (and fallback to OCR for other images).
    Returns data as a list of dicts with meta info, ready for vectorization.
    """
    data = []
    ext = os.path.splitext(file_path)[-1].lower()

    # PDF
    if ext == ".pdf":
        text_pages = extract_text_from_pdf(file_path)
        for item in text_pages:
            page = item["page"]
            for i, para in enumerate(item["text"].split('\n\n')):
                para = para.strip()
                if para:
                    data.append({
                        "id": doc_id,
                        "doc_name": doc_name,
                        "page": page,
                        "para": i + 1,
                        "text": para
                    })
    # Image types (jpg, jpeg, png)
    elif ext in [".jpg", ".jpeg", ".png"]:
        text = extract_text_from_image(file_path)
        for i, para in enumerate(text.split('\n\n')):
            para = para.strip()
            if para:
                data.append({
                    "id": doc_id,
                    "doc_name": doc_name,
                    "page": 1,
                    "para": i + 1,
                    "text": para
                })
    # TXT
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
        for i, para in enumerate(text.split('\n\n')):
            para = para.strip()
            if para:
                data.append({
                    "id": doc_id,
                    "doc_name": doc_name,
                    "page": 1,
                    "para": i + 1,
                    "text": para
                })
    # DOCX
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        for i, para in enumerate(text.split('\n\n')):
            para = para.strip()
            if para:
                data.append({
                    "id": doc_id,
                    "doc_name": doc_name,
                    "page": 1,
                    "para": i + 1,
                    "text": para
                })
    else:
        # fallback: treat as image for OCR
        try:
            text = extract_text_from_image(file_path)
            for i, para in enumerate(text.split('\n\n')):
                para = para.strip()
                if para:
                    data.append({
                        "id": doc_id,
                        "doc_name": doc_name,
                        "page": 1,
                        "para": i + 1,
                        "text": para
                    })
        except Exception:
            raise ValueError(f"Unsupported file type or unable to process {file_path}")

    return data


def get_embedding(text):
    # This already returns a list in your code, but let's be sure.
    emb = list(_embedder.embed([text]))[0]
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    return emb

def upsert_to_pinecone(split_data, index_name):
    index = _pc.Index(index_name)
    vectors = []
    for chunk in split_data:
        text = chunk["text"]
        embedding = get_embedding(text)
        unique_id = f"{chunk['id']}_{chunk['page']}_{chunk['para']}"
        meta = {
            "doc_name": chunk.get("doc_name"),
            "page": chunk.get("page"),
            "para": chunk.get("para"),
            "text": text
        }
        vectors.append({
            "id": unique_id,
            "values": embedding,
            "metadata": meta
        })
    # Upsert in batches if large
    index.upsert(vectors=vectors)

def delete_index(index_name):
    _pc.delete_index(index_name)
