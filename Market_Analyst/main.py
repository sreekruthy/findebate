import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure correct working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag import initialize_rag
from market_agent.market_agent import market_analyst


def extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON found in response")

    return json.loads(match.group())


# Initialize RAG
initialize_rag()

# Run agent
tesla_raw = market_analyst("Tesla")
apple_raw = market_analyst("Apple")

# Parse JSON safely
tesla_json = extract_json(tesla_raw)
apple_json = extract_json(apple_raw)

# Professional output
output = {
    "project": {
        "name": "AI Financial Analyst",
        "type": "RAG + Gemini LLM",
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

with open("output.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Final output saved to output.json")