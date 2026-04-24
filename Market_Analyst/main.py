import os
import json
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag import initialize_rag
from market_agent.market_agent import market_analyst

# Initialize RAG
initialize_rag()

# Run Gemini agent
tesla = market_analyst("Tesla")
apple = market_analyst("Apple")

# 🔍 DEBUG (print BEFORE parsing)
print("\n--- TESLA RAW OUTPUT ---\n", tesla)
print("\n--- APPLE RAW OUTPUT ---\n", apple)


# ✅ Safe JSON extractor
def extract_json(text):
    if not text:
        raise ValueError("Empty response from LLM")

    # Remove markdown formatting
    text = text.replace("```json", "").replace("```", "").strip()

    # Extract JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print("❌ JSON parsing error:", e)
            print("RAW OUTPUT:\n", text)
            raise
    else:
        print("❌ No JSON found")
        print("RAW OUTPUT:\n", text)
        raise ValueError("No JSON found")


# Convert safely
tesla_json = extract_json(tesla)
apple_json = extract_json(apple)

# Final output
output = {
    "model": "Market Analyst Agent (Gemini + RAG)",
    "analysis": [tesla_json, apple_json]
}

# Save file
with open("output.json", "w") as f:
    json.dump(output, f, indent=4)

print("\n✅ Gemini Output saved to output.json")