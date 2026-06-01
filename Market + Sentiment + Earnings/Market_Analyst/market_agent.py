import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from MVP.rag_module import initialize_rag, retrieve_filtered

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── LEVEL 1 — SYSTEM PROMPT (professional identity) ──
MARKET_SYSTEM_PROMPT = """You are a senior quantitative strategist and former portfolio manager \
with extensive experience in institutional market timing and systematic trading strategies \
at firms including Two Sigma, Point72, and Millennium Management.

Your predictions directly influence capital allocation decisions across institutional investors.

PROFESSIONAL STANDARDS:
- Support all predictions with specific content from the actual source documents
- Maintain realistic confidence levels (70-80%) rather than overconfident assertions
- Avoid speculative predictions not grounded in actual business fundamentals
- Deliver output in strict JSON format — no markdown, no preamble, no extra text"""


# ── LEVEL 2 — USER PROMPT (3-horizon framework + RAG context) ──
def build_market_user_prompt(source_file: str, context: str) -> str:
    return f"""Perform multi-timeframe market analysis for {source_file} using ONLY the context below.

SYSTEMATIC MULTI-TIMEFRAME FRAMEWORK:
1. 1-Day Horizon — immediate earnings reaction signal (beat/miss, management tone, catalysts)
2. 1-Week Horizon — momentum drivers sustaining weekly performance
3. 1-Month Horizon — fundamental positioning and guidance trajectory

Final composite score:
- Strong bullish signals → 7-9
- Mixed/neutral signals  → 4-6
- Bearish signals        → 1-3

Return ONLY valid JSON, no markdown, no extra text:
{{
  "agent": "Market Analyst",
  "source_file": "{source_file}",
  "key_points": [
    "1-day market reaction signal max 25 words",
    "1-week momentum driver max 25 words",
    "1-month fundamental positioning max 25 words",
    "Key risk or macro factor max 25 words"
  ],
  "score": <float 0.0-10.0>,
  "reasoning": "2-3 sentences covering all three horizons with specific evidence from context"
}}

Context:
{context}"""


# ── RAG RETRIEVAL ──
def get_market_context(company: str) -> str:
    query = "market trends stock movement industry outlook analyst sentiment earnings surprise"
    results = retrieve_filtered(query=query, company=company, data_type="news_q1", k=4)
    if not results:
        return "No market data available."
    return "\n\n".join(f"[Chunk {i+1}]: {chunk.strip()}" for i, chunk in enumerate(results))


# ── JSON EXTRACTOR ──
def extract_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in market agent response")
    parsed = json.loads(match.group())
    if "score" in parsed:
        parsed["score"] = round(min(float(parsed["score"]), 10.0), 1)
    return parsed


# ── GROQ CALL (2-level: system + user) ──
def market_analyst(company: str) -> str:
    context = get_market_context(company)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": MARKET_SYSTEM_PROMPT},
            {"role": "user",   "content": build_market_user_prompt(company, context)}
        ],
        temperature=0.3,
        max_tokens=1200
    )
    return response.choices[0].message.content.strip()


# ── MAIN ──
def main():
    initialize_rag()
    tesla_raw  = market_analyst("Tesla")
    apple_raw  = market_analyst("Apple")
    tesla_json = extract_json(tesla_raw)
    apple_json = extract_json(apple_raw)

    output = {
        "project": {
            "type": "Market Agent",
            "description": "Multi-timeframe market analysis with 2-level prompt structure (Section 2.2 FinDebate)"
        },
        "generated_at": datetime.now().isoformat(),
        "companies": ["Tesla", "Apple"],
        "analysis": [tesla_json, apple_json],
        "comparison": {
            "better_market_signal": (
                tesla_json["company"] if tesla_json["score"] > apple_json["score"]
                else apple_json["company"]
            ),
            "score_gap": round(abs(tesla_json["score"] - apple_json["score"]), 2)
        }
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/market_output.json", "w") as f:
        json.dump(output, f, indent=4)
    print("✅ Market output saved → outputs/market_output.json")


if __name__ == "__main__":
    main()