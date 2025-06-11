import requests
from ..config import settings

def gemini_chat(messages, temperature=0.2):
    """
    Calls Gemini API for generating chat responses.
    
    Args:
        messages (list[str]): Conversation history as a list of message strings.
        temperature (float): Sampling temperature for generation.

    Returns:
        str: Generated response text.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": msg} for msg in messages]}],
        "generationConfig": {"temperature": temperature}
    }
    params = {"key": settings.GEMINI_API_KEY}
    
    # Make a POST request to the Gemini API
    resp = requests.post(url, headers=headers, params=params, json=data)
    resp.raise_for_status()  # Raises exception for HTTP errors
    
    # Return the generated text from response
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
