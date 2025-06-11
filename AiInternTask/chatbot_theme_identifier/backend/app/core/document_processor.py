import os
import uuid
import requests
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
from ..config import settings, _embedder, _pc
from docx import Document

def extract_text_from_txt(txt_path):
    """Extract full text from a .txt file."""
    with open(txt_path, encoding='utf-8') as f:
        return f.read()

def extract_text_from_docx(docx_path):
    """
    Extracts text from a .docx file, handling manual page breaks.
    Returns a list of dicts: {"page": <page_number>, "text": <page_text>}
    """
    doc = Document(docx_path)
    text = []
    current_page_text = []
    page_number = 1

    for para in doc.paragraphs:
        # Detect manual page break in paragraph runs
        has_page_break = any('pageBreak' in run._element.xml for run in para.runs)
        if has_page_break and current_page_text:
            # Save current page, increment, and start a new one
            text.append({"page": page_number, "text": "\n".join(current_page_text)})
            page_number += 1
            current_page_text = []
        if para.text.strip():
            current_page_text.append(para.text.strip())
    if current_page_text:
        text.append({"page": page_number, "text": "\n".join(current_page_text)})
    return text

def extract_text_from_pdf(file_path):
    """Extracts text from each page of a PDF."""
    reader = PdfReader(file_path)
    text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text.append({"page": i + 1, "text": page_text})
    return text

def extract_text_from_image(image_path):
    """Extracts text from an image using OCR (pytesseract)."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def process_and_split_document(file_path, doc_name, doc_id):
    """
    Process a document (PDF, image, txt, docx), extract and split into paragraphs.
    Returns a list of dicts for vectorization/upsert.
    """
    data = []
    ext = os.path.splitext(file_path)[-1].lower()

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
    elif ext in [".jpg", ".jpeg", ".png"]:
        # OCR for image files
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
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        # For .docx, output from extract_text_from_docx is a list of dicts
        for item in text:
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
    else:
        # Fallback: try OCR for any other file type
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
    """Generate embedding vector for given text."""
    emb = list(_embedder.embed([text]))[0]
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    return emb

def upsert_to_pinecone(split_data, index_name):
    """
    Upserts a list of text chunks (with metadata) into Pinecone index.
    """
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
    index.upsert(vectors=vectors)

def delete_index(index_name):
    """Deletes the Pinecone index with the specified name."""
    _pc.delete_index(index_name)
