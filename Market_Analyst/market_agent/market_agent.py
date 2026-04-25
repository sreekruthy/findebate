import os
from google import genai
from rag import retrieve_filtered

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def market_analyst(company):

    documents = retrieve_filtered(
        query="market trends, stock movement, industry outlook",
        company=company,
        data_type="news_q1"
    )

    context = " ".join(documents)

    prompt = f"""
You are a financial market analyst AI.

Analyze the data and return STRICT JSON only.

Rules:
- No explanations
- No markdown
- Only valid JSON

Format:
{{
    "company": "{company}",
    "agent": "Market Analyst",
    "key_points": ["...", "...", "..."],
    "score": number,
    "reasoning": "..."
}}

Data:
{context}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text