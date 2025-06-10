import os
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from ..config import EMBEDDING_MODEL, CHROMA_COLLECTION

chroma_client = chromadb.Client(Settings())
chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
embedder = SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(file_path):
    text_chunks = []
    reader = PdfReader(file_path)
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                for para_num, para in enumerate(text.split('\n\n')):
                    if para.strip():
                        text_chunks.append({
                            "page": page_num + 1,
                            "para": para_num + 1,
                            "text": para.strip()
                        })
        except Exception as e:
            print(f"Page {page_num+1} failed: {e}")
    return text_chunks

def extract_text_from_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return [{
        "page": 1,
        "para": 1,
        "text": text.strip()
    }] if text.strip() else []

def add_document_to_db(doc_id, file_name, chunks):
    for chunk in chunks:
        emb = embedder.encode(chunk['text'])
        chunk_id = f"{doc_id}_p{chunk['page']}_para{chunk['para']}"
        chroma_collection.add(
            embeddings=[emb],
            documents=[chunk['text']],
            metadatas=[{
                "doc_id": doc_id,
                "file_name": file_name,
                "page": chunk["page"],
                "para": chunk["para"]
            }],
            ids=[chunk_id],
        )

def process_uploaded_file(file_path, doc_id, file_name):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".pdf"]:
        chunks = extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        chunks = extract_text_from_image(file_path)
    elif ext in [".txt"]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = [{
            "page": 1,
            "para": i+1,
            "text": para.strip()
        } for i, para in enumerate(text.split('\n\n')) if para.strip()]
    else:
        raise Exception("Unsupported file type.")
    add_document_to_db(doc_id, file_name, chunks)
    return len(chunks)
