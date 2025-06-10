from sentence_transformers import SentenceTransformer
from ..config import EMBEDDING_MODEL, CHROMA_COLLECTION
import chromadb
from ..services.gemini_service import gemini_chat

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
embedder = SentenceTransformer(EMBEDDING_MODEL)

def retrieve_relevant_chunks(user_query, top_k=10):
    query_emb = embedder.encode(user_query)
    results = chroma_collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=['documents', 'metadatas']
    )
    output = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        output.append({
            "doc_id": meta["doc_id"],
            'file_name': meta["file_name"],
            "page": meta["page"],
            "para": meta["para"],
            "text": doc
        })
    return output

def extract_answers(user_query, top_chunks):
    per_doc_answers = []
    for chunk in top_chunks:
        prompt = f"""Given the following context from document {chunk['doc_id']} (Page {chunk['page']}, Paragraph {chunk['para']}):
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
            "file_name": chunk['file_name'],
            "answer": answer,
            "citation": f"Page {chunk['page']}, Para {chunk['para']}"
        })
    return per_doc_answers
