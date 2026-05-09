# market_agent.py — Person 3 | Group A | Module 2 (Earnings Analyst + Market Predictor)
# Implements the 2-level prompt structure from FinDebate Section 2.2:
#   Level 1 — System Prompt : professional identity (credentials, background, mission, quality standard)
#   Level 2 — User Prompt  : analytical task (framework, technical requirements, output spec, RAG context)

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from rag_module import initialize_rag, retrieve_filtered

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ──────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — SYSTEM PROMPT  (professional identity)
# ──────────────────────────────────────────────────────────────────────────────
MARKET_SYSTEM_PROMPT = """You are a senior quantitative strategist and former portfolio manager \
with extensive experience in institutional market timing and systematic trading strategies \
at firms including Two Sigma, Point72, and Millennium Management.

Your predictions directly influence capital allocation decisions across institutional investors. \
Professional portfolio managers will execute trades based on your systematic market timing \
analysis grounded in actual earnings call and financial news content.

INSTITUTIONAL MARKET TIMING AUTHORITY:
Deliver high-conviction market predictions with the precision required for institutional \
trading decisions, but maintain realistic confidence levels (70–80%) and base all \
assessments on actual earnings call and news content rather than hypothetical scenarios \
or unverifiable market data.

PROFESSIONAL STANDARDS:
- Support all predictions with specific content from the actual source documents
- Maintain realistic confidence levels (70–80%) rather than overconfident assertions
- Avoid speculative market timing predictions not grounded in actual business fundamentals
- Focus on institutional factors that can be derived from actual management commentary
- Deliver output in strict JSON format — no markdown, no preamble, no extra text"""


# ──────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — USER PROMPT  (analytical task + RAG context)
# ──────────────────────────────────────────────────────────────────────────────
MARKET_USER_PROMPT = """ANALYTICAL TASK — MULTI-TIMEFRAME MARKET ANALYSIS FOR {company}

SYSTEMATIC MULTI-TIMEFRAME FRAMEWORK (evaluate all three horizons):

1. IMMEDIATE MARKET REACTION (1-Day Horizon)
   - Earnings surprise assessment (beat / miss vs. stated expectations)
   - Management tone and confidence as demonstrated in the actual call / news
   - Specific positive or negative catalysts mentioned in the source documents

2. MOMENTUM ANALYSIS (1-Week Horizon)
   - Fundamental drivers sustaining weekly price momentum
   - Analyst and investor sentiment signals from the Q&A or news
   - Near-term catalysts or headwinds mentioned by management

3. FUNDAMENTAL POSITIONING (1-Month Horizon)
   - Business fundamental trajectory over the medium term
   - Strategic developments and management guidance for next quarter
   - Macro or sector-level risks disclosed in the source documents

TECHNICAL REQUIREMENTS:
- For each horizon assign an internal score 0–10 (DO NOT output these)
- Combine into ONE final composite market score:
    Strong bullish signals across all horizons → 7–9
    Mixed / neutral signals                   → 4–6
    Bearish signals or high uncertainty       → 1–3
- Confidence calibration: 70–80%; avoid overconfidence
- Only cite information present in the context below
- Each key point must be ≤ 25 words

OUTPUT SPECIFICATION — return ONLY valid JSON, no markdown, no extra text:

{{
  "agent": "Market Analyst",
  "company": "{company}",
  "source_file": "clean_news1.txt / clean_news2.txt",
  "key_points": [
    "1-day market reaction signal ≤25 words",
    "1-week momentum driver ≤25 words",
    "1-month fundamental positioning ≤25 words",
    "Key risk or macro factor ≤25 words"
  ],
  "score": <float 0.0–10.0>,
  "reasoning": "2–3 sentence justification covering all three horizons and citing specific evidence from the news/earnings context"
}}

CONTEXTUAL INTEGRATION (RAG-retrieved evidence — use ONLY this):
---------------------
{context}
---------------------"""


# ──────────────────────────────────────────────────────────────────────────────
# RAG retrieval
# ──────────────────────────────────────────────────────────────────────────────
def get_market_context(company: str) -> str:
    """Retrieve market / news chunks via domain-specific RAG."""
    query = (
        "market trends stock movement industry outlook analyst sentiment "
        "investor reaction earnings surprise guidance upgrade downgrade"
    )
    results = retrieve_filtered(query=query, company=company, data_type="news_q1", k=4)
    if not results:
        return "No market data available."
    return "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {chunk.strip()}" for i, chunk in enumerate(results)
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON extractor
# ──────────────────────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in market agent response")
    parsed = json.loads(match.group())
    if "score" in parsed:
        parsed["score"] = round(min(float(parsed["score"]), 10.0), 1)
    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# Gemini call — 2-level prompt structure
# ──────────────────────────────────────────────────────────────────────────────
def run_gemini_market(company: str, context: str) -> str:
    user_prompt = MARKET_USER_PROMPT.replace("{company}", company).replace("{context}", context)

    # 2-level structure: system + user combined (Gemini does not support separate system role)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {"role": "user", "parts": [{"text": MARKET_SYSTEM_PROMPT + "\n\n" + user_prompt}]}
        ],
    )
    return response.text


# ──────────────────────────────────────────────────────────────────────────────
# Main agent entry point
# ──────────────────────────────────────────────────────────────────────────────
def market_analyst(company: str) -> str:
    """Run the Market Analyst agent and return raw model text (parse with extract_json)."""
    context = get_market_context(company)
    return run_gemini_market(company, context)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test — runs both companies and saves comparison output
# ──────────────────────────────────────────────────────────────────────────────
def main():
    initialize_rag()

    tesla_raw = market_analyst("Tesla")
    apple_raw = market_analyst("Apple")

    tesla_json = extract_json(tesla_raw)
    apple_json = extract_json(apple_raw)

    output = {
        "project": {
            "type": "Market Agent",
            "description": (
                "Multi-timeframe market analysis using RAG-retrieved financial news "
                "and LLM reasoning with 2-level prompt structure (Section 2.2 FinDebate)"
            ),
        },
        "generated_at": datetime.now().isoformat(),
        "companies": ["Tesla", "Apple"],
        "analysis": [tesla_json, apple_json],
        "comparison": {
            "better_market_signal": (
                tesla_json["company"]
                if tesla_json["score"] > apple_json["score"]
                else apple_json["company"]
            ),
            "score_gap": round(abs(tesla_json["score"] - apple_json["score"]), 2),
        },
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/market_output.json", "w") as f:
        json.dump(output, f, indent=4)
    print("✅ Market output saved → outputs/market_output.json")


if __name__ == "__main__":
    main()