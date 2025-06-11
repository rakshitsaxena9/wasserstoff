from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import endpoints

# Create FastAPI application instance
app = FastAPI()

# Include API endpoints router
app.include_router(endpoints.router)

# Allow CORS for all origins and methods (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
