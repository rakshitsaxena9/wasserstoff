import os
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pinecone import Pinecone

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env')))

class Settings:
    HF_API_KEY = os.getenv("HF_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_HOST = os.getenv("PINECONE_HOST")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

settings = Settings()


# Load the embedder ONCE globally for efficiency (small RAM use)
_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  
EMBEDDING_DIM = 384
_pc = Pinecone(api_key=settings.PINECONE_API_KEY)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
for key in ["GEMINI_API_KEY", "HF_API_KEY", "PINECONE_API_KEY", "PINECONE_HOST"]:
    if not getattr(settings, key, None):
        raise ValueError(f"{key} is not set in your .env file.")