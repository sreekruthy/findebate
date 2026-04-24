from rag import retrieve_filtered
from google import genai

client = genai.Client(api_key="AIzaSyBIA1XcXLMmDqUdAzpnnwCX5-psLPpfWQY")

def market_analyst(company):

    # Step 1: Retrieve data from RAG
    documents = retrieve_filtered(
        query="market trends, stock movement, industry outlook",
        company=company,
        data_type="news_q1"
    )

    context = " ".join(documents)

    # Step 2: Prompt for Gemini
    prompt = f"""
    You are a financial market analyst AI.

    Analyze the following company data and return:
    - key_points (max 3)
    - score (0–10)
    - reasoning

    Company: {company}

    Data:
    {context}

    Return output ONLY in JSON format like:
    {{
        "company": "{company}",
        "agent": "Market Analyst",
        "key_points": [...],
        "score": number,
        "reasoning": "..."
    }}
    """

    # Step 3: Call Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Step 4: Return result
    return response.text