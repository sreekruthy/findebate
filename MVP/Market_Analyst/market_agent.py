import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from rag_module import initialize_rag, retrieve_filtered

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# =========================
# MARKET ANALYST FUNCTION
# =========================
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
    "score": <float between 0.0 and 10.0 only>,
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


# =========================
# JSON EXTRACTOR
# =========================
def extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON found in response")

    return json.loads(match.group())


# =========================
# MAIN EXECUTION
# =========================
def main():

    # Initialize RAG
    initialize_rag()

    # Run analysis
    tesla_raw = market_analyst("Tesla")
    apple_raw = market_analyst("Apple")

    # Parse results
    tesla_json = extract_json(tesla_raw)
    apple_json = extract_json(apple_raw)

    # Final structured output
    output = {
        "project": {
            "type": "Market Agent",
            "description": "Market analysis using retrieved financial data and LLM reasoning"
        },
        "generated_at": datetime.now().isoformat(),
        "companies": ["Tesla", "Apple"],
        "analysis": [tesla_json, apple_json],
        "comparison": {
            "better_company": (
                tesla_json["company"]
                if tesla_json["score"] > apple_json["score"]
                else apple_json["company"]
            ),
            "score_gap": abs(tesla_json["score"] - apple_json["score"])
        }
    }

    # Save output
    with open("output.json", "w") as f:
        json.dump(output, f, indent=4)

    print("✅ Final output saved to output.json")


# =========================
# RUN PROGRAM
# =========================
if __name__ == "__main__":
    main()