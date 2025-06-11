import requests
from ..config import settings

def gemini_chat(messages, temperature=0.2):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": msg} for msg in messages]}],
        "generationConfig": {"temperature": temperature}
    }
    params = {"key": settings.GEMINI_API_KEY}
    resp = requests.post(url, headers=headers, params=params, json=data)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
