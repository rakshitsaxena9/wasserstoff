from pydantic import BaseModel

class QueryRequest(BaseModel):
    user_query: str

class UploadResponse(BaseModel):
    success: bool
    doc_id: str = None
    n_chunks: int = None
    error: str = None

class QueryResponse(BaseModel):
    answers: list
    themes: str
