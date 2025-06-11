from pydantic import BaseModel

class QueryRequest(BaseModel):
    user_query: str  # User's query string

class UploadResponse(BaseModel):
    success: bool               # Indicates if upload was successful
    doc_id: str = None          # Optional: Document ID
    n_chunks: int = None        # Optional: Number of chunks processed
    error: str = None           # Optional: Error message, if any

class QueryResponse(BaseModel):
    answers: list   # List of answers returned from query
    themes: str     # Synthesized themes string
