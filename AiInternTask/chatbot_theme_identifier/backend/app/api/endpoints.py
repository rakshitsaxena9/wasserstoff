import os
import uuid
import shutil

from fastapi import APIRouter, UploadFile, File, Form
from ..core.document_processor import process_uploaded_file, chroma_collection
from ..core.query_pipeline import retrieve_relevant_chunks, extract_answers
from ..core.theme_synthesis import synthesize_themes
from ..config import UPLOAD_DIR

router = APIRouter()

@router.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    # Check if file_name already exists in vector DB
    existing = chroma_collection.get(where={"file_name": {"$eq": file.filename}})
    if existing and len(existing['ids']) > 0:
        return {"success": False, "error": f"Document '{file.filename}' already uploaded."}

    doc_id = str(uuid.uuid4())[:8]
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        n_chunks = process_uploaded_file(file_path, doc_id, file.filename)
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success": True, "doc_id": doc_id, "n_chunks": n_chunks}

def deduplicate_answers(per_doc_answers):
    seen = set()
    deduped = []
    for ans in per_doc_answers:
        key = (ans.get("file_name", ans.get("doc_id")), ans["answer"].strip().lower())
        if key not in seen:
            deduped.append(ans)
            seen.add(key)
    return deduped

@router.post("/query/")
async def query_docs(user_query: str = Form(...)):
    top_chunks = retrieve_relevant_chunks(user_query)
    per_doc_answers = extract_answers(user_query, top_chunks)
    per_doc_answers = deduplicate_answers(per_doc_answers)
    themes = synthesize_themes(user_query, per_doc_answers)
    return {"answers": per_doc_answers, "themes": themes}
