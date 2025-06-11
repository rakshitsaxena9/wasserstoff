from ..config import settings, _embedder, _pc
from ..services.gemini_service import gemini_chat
from .document_processor import get_embedding

def pinecone_query(embedding, index_name, top_k=10):
    index = _pc.Index(index_name)
    query_results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return query_results.get("matches", [])

def retrieve_relevant_docs(question: str, index_name: str, top_k: int = 10):
    embedding = get_embedding(question)
    matches = pinecone_query(embedding, index_name=index_name, top_k=top_k)
    return matches

def build_citation_table(matches: list[dict]) -> list[dict]:
    table = []
    for m in matches:
        meta = m.get("metadata", {})
        table.append({
            "doc_name": meta.get("doc_name"),
            "para": meta.get("para"),
            "page": meta.get("page"),
            "text": meta.get("text"),
            "score": m.get("score", 0),
            "doc_id": m.get("id")
        })
    return table

def extract_answers(user_query, top_chunks):
    per_doc_answers = []
    for chunk in top_chunks:
        prompt = f"""Given the following context from document {chunk['doc_name']} (Page {chunk['page']}, Paragraph {chunk['para']}):
-------------------
{chunk['text']}
-------------------
Answer the question: "{user_query}" in a concise sentence, citing the document/page/para.
"""
        try:
            answer = gemini_chat([prompt])
        except Exception as e:
            answer = f"LLM API failed: {e}"
            continue
        if not answer or "does not specify" in answer.lower() or "doesn't provide"  in answer.lower() or "not provide"  in answer.lower() or "doesn't specify" in answer.lower() or "doesn't mention" in answer.lower()  or "not mention" in answer.lower() or "cannot answer" in answer.lower():
            continue  # Skip unhelpful answers
        per_doc_answers.append({
            "doc_id": chunk['doc_id'],
            "doc_name": chunk['doc_name'],
            "answer": answer,
            "citation": f"Page {chunk['page']}, Para {chunk['para']}"
        })
    return per_doc_answers
