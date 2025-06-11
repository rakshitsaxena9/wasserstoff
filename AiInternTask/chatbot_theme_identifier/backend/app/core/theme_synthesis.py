from ..services.gemini_service import gemini_chat

def synthesize_themes(user_query, per_doc_answers):
    """
    Synthesizes key themes from multiple per-document answers using an LLM.
    Returns a concise summary with document citations.
    """
    if not per_doc_answers:
        return "Not enough context in the uploaded documents to answer this question."

    formatted = "\n".join(
        [f"Document {a['doc_name']} ({a['citation']}): {a['answer']}" for a in per_doc_answers]
    )
    prompt = f"""
Given these answers from various documents for the question: "{user_query}":
{formatted}

Identify the main themes (1-4) present in these answers. For each theme, provide:
- A short title (e.g., "Theme 1 - Regulatory Non-Compliance")
- A concise, consolidated summary (2-4 sentences) addressing that theme, using evidence from the supporting documents.
- Include supporting document names or IDs and their citations.
- Be concise, avoid repetition, and present each theme clearly.

Format:
Theme 1 - <Short Title>:

 <Citation Document Names>: 
 <Consolidated answer> 

Theme 2 - <Short Title>:

 <Citation Document Names>:
 <Consolidated answer> 

If there is only one clear theme, output just one theme.
    """
    try:
        response = gemini_chat([prompt])
    except Exception as e:
        response = f"Theme synthesis failed: {e}"
    return response
