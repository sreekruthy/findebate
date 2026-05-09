# earnings_agent.py — Person 3 | Group A | Module 2 (Earnings Analyst + Market Predictor)
# Implements the 2-level prompt structure from FinDebate Section 2.2:
#   Level 1 — System Prompt : professional identity (credentials, background, mission, quality standard)
#   Level 2 — User Prompt  : analytical task (framework, technical requirements, output spec, RAG context)

import os
import json
from rag_module import initialize_rag, retrieve_filtered
from dotenv import load_dotenv

load_dotenv()

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — SYSTEM PROMPT  (professional identity)
# ──────────────────────────────────────────────────────────────────────────────
EARNINGS_SYSTEM_PROMPT = """You are a CFA charterholder and senior equity research analyst \
with 20+ years of experience analyzing financial statements for premier investment banks \
and hedge funds including Goldman Sachs, Morgan Stanley, and Citadel.

Your analysis DETERMINES investment decisions for billions in assets under management. \
Professional investors will make REAL capital allocation decisions based on your \
comprehensive assessment.

INSTITUTIONAL AUTHORITY MISSION:
Deliver definitive, data-driven earnings analysis with the depth and precision expected \
by institutional investment committees. Your assessment must be comprehensive enough to \
support major portfolio allocation decisions and provide clear directional conviction \
with supporting evidence based STRICTLY on the actual earnings call content provided.

PROFESSIONAL STANDARDS:
- Base all assessments on verifiable information from the actual earnings call
- Maintain realistic confidence levels (70-80%) rather than overconfident assertions
- Focus on management's actual explanations rather than hypothetical scenarios
- Never hallucinate numbers not present in the source material
- Deliver institutional-grade output in strict JSON format — no markdown, no preamble"""


# ──────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — USER PROMPT  (analytical task + RAG context)
# ──────────────────────────────────────────────────────────────────────────────
EARNINGS_USER_PROMPT = """ANALYTICAL TASK — EARNINGS ANALYSIS FOR {company}

ANALYTICAL FRAMEWORK (evaluate internally on 4 dimensions):

1. Revenue Performance
   - Growth strength and year-over-year momentum
   - Segment-level breakdown (if discussed by management)
   - Beat / miss versus stated expectations

2. Profitability
   - Gross margin, operating margin trends
   - EPS growth and earnings quality

3. Earnings Quality & Sustainability
   - Recurring vs one-time components
   - Cash flow strength relative to reported income

4. Management Guidance
   - Forward revenue and margin guidance
   - Confidence signals and risk disclosures in Q&A

TECHNICAL REQUIREMENTS:
- For each dimension assign an internal score 0–10 (DO NOT output these)
- Combine into ONE final score using balanced judgment:
    Strong across all dimensions → 8–9
    Mixed signals               → 5–7
    Weak fundamentals           → 2–4
- Only cite numbers explicitly present in the context below
- Each key point must be ≤ 25 words
- Confidence calibration: 70–80% range; avoid overconfidence

OUTPUT SPECIFICATION — return ONLY valid JSON, no markdown, no extra text:

{{
  "agent": "Earnings Analyst",
  "company": "{company}",
  "source_file": "clean_clean_clean_earnings_q1.txt",
  "key_points": [
    "Revenue insight ≤25 words",
    "Profitability insight ≤25 words",
    "Earnings quality insight ≤25 words",
    "Management guidance insight ≤25 words"
  ],
  "score": <float 0.0–10.0>,
  "reasoning": "2–3 sentence justification citing the four dimensions and referencing specific figures from the earnings call"
}}

CONTEXTUAL INTEGRATION (RAG-retrieved evidence — use ONLY this):
---------------------
{context}
---------------------"""


# ──────────────────────────────────────────────────────────────────────────────
# RAG retrieval
# ──────────────────────────────────────────────────────────────────────────────
def get_earnings_context(company: str) -> str:
    """Retrieve earnings-specific chunks via domain-specific RAG."""
    query = (
        "revenue growth profitability margins earnings guidance "
        "financial performance EPS net income gross margin"
    )
    results = retrieve_filtered(query=query, company=company, data_type="earnings_q1", k=4)
    if not results:
        return "No financial data available."
    return "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {chunk.strip()}" for i, chunk in enumerate(results)
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON safety parser
# ──────────────────────────────────────────────────────────────────────────────
def safe_parse(output: str) -> dict:
    try:
        return json.loads(output)
    except Exception:
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            return json.loads(output[start:end])
        except Exception:
            return {"error": "Invalid JSON", "raw": output}


# ──────────────────────────────────────────────────────────────────────────────
# Gemini call — 2-level prompt structure
# ──────────────────────────────────────────────────────────────────────────────
def run_gemini_earnings(company: str, context: str) -> str:
    user_prompt = EARNINGS_USER_PROMPT.replace("{company}", company).replace("{context}", context)

    # 2-level structure: system content + user content sent as a conversation
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {"role": "user", "parts": [{"text": EARNINGS_SYSTEM_PROMPT + "\n\n" + user_prompt}]}
        ],
    )
    return response.text


# ──────────────────────────────────────────────────────────────────────────────
# Main agent entry point
# ──────────────────────────────────────────────────────────────────────────────
def earnings_agent(company: str) -> dict:
    """Run the Earnings Analyst agent for a given company and return structured JSON."""
    context = get_earnings_context(company)
    raw_output = run_gemini_earnings(company, context)
    parsed = safe_parse(raw_output)
    # Clamp score
    if "score" in parsed and not isinstance(parsed.get("score"), str):
        parsed["score"] = round(min(float(parsed["score"]), 10.0), 1)
    return parsed


def save_output(result: dict, company: str) -> None:
    filename = f"outputs/{company.lower()}_earnings.json"
    os.makedirs("outputs", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Earnings output saved → {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize_rag()
    for company in ["Apple", "Tesla"]:
        result = earnings_agent(company)
        print(json.dumps(result, indent=2))
        save_output(result, company)