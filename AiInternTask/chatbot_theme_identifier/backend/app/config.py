import os
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pinecone import Pinecone

# Load environment variables from .env file two directories up
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env')))

class Settings:
    # Store sensitive API keys from environment
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

settings = Settings()

# Load embedding model globally (efficient memory use)
_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384

# Initialize Pinecone client globally
_pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Directory to store uploaded documents
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Ensure required keys are set; raise error early if missing
for key in ["GEMINI_API_KEY",  "PINECONE_API_KEY"]:
    if not getattr(settings, key, None):
        raise ValueError(f"{key} is not set in your .env file.")
