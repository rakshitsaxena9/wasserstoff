import os
from dotenv import load_dotenv

# Load variables from .env file at project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', '.env'))

# Now use environment variables as before
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
CHROMA_COLLECTION = "documents"
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")
