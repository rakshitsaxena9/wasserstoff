import os
import uuid
import shutil

from fastapi import APIRouter, UploadFile, File, Form
from ..core.document_processor import process_and_split_document, upsert_to_pinecone, delete_index
from ..core.query_pipeline import retrieve_relevant_docs, build_citation_table, extract_answers
from ..core.theme_synthesis import synthesize_themes
from ..config import UPLOAD_DIR, settings, _pc, EMBEDDING_DIM
import requests
import tempfile
from pinecone import ServerlessSpec

router = APIRouter()

def pinecone_check_index_exists(index_name):
    return index_name in [idx.name for idx in _pc.list_indexes()]

@router.post("/upload/")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...)  # require client to send same session_id for all uploads in session
):
    print("Got the File")
    index_name = f"wasserstoff-{session_id}"

    # Create index for the session if it doesn't exist
    if not pinecone_check_index_exists(index_name):
        _pc.create_index(
            name=index_name,
            dimension= EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
              )
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        tmp_file.write(file.file.read())  # read() because UploadFile is already a file-like
        tmp_file_path = tmp_file.name
        print("File Opened")
    try:
        chunks = process_and_split_document(tmp_file_path, file.filename, session_id)
        print("Processing Completed..")
        upsert_to_pinecone(chunks, index_name=index_name)
        n_chunks = len(chunks)
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            os.remove(tmp_file_path)
        except Exception as e:
            print(f"Could not remove temporary file: {e}")
    return {
        "success": True,
        "session_id": session_id,
        "index": index_name,
        "n_chunks": n_chunks
    }

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
async def query_docs(
    user_query: str = Form(...),
    session_id: str = Form(...)
):
    index_name = f"wasserstoff-{session_id}"
    matches = retrieve_relevant_docs(user_query, index_name=index_name)
    table = build_citation_table(matches)
    per_doc_answers = extract_answers(user_query, table)
    per_doc_answers = deduplicate_answers(per_doc_answers)
    themes = synthesize_themes(user_query, per_doc_answers)
    return {"answers": per_doc_answers, "themes": themes}

@router.delete("/delete/")
async def delete_session(session_id: str = Form(...)):
    index_name = f"wasserstoff-{session_id}"
    try:
        delete_index(index_name)
        return {"success": True, "message": f"Index {index_name} deleted."}
    except Exception as e:
        return {"success": False, "error": str(e)}