# earnings_agent.py — Person 3 | Group A | Module 2
# 2-level prompt structure per FinDebate Section 2.2
# Level 1 — System Prompt: professional identity
# Level 2 — User Prompt: analytical task + RAG context

import os
import json
from rag_module import initialize_rag, retrieve_filtered
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── LEVEL 1 — SYSTEM PROMPT (professional identity) ──
EARNINGS_SYSTEM_PROMPT = """You are a CFA charterholder and senior equity research analyst \
with 20+ years of experience analyzing financial statements for premier investment banks \
and hedge funds including Goldman Sachs, Morgan Stanley, and Citadel.

Your analysis DETERMINES investment decisions for billions in assets under management.
Professional investors will make REAL capital allocation decisions based on your assessment.

PROFESSIONAL STANDARDS:
- Base all assessments on verifiable information from the actual earnings call
- Maintain realistic confidence levels (70-80%) rather than overconfident assertions
- Never hallucinate numbers not present in the source material
- Deliver institutional-grade output in strict JSON format — no markdown, no preamble"""


# ── LEVEL 2 — USER PROMPT (analytical task + RAG context) ──
def build_earnings_user_prompt(source_file: str, context: str) -> str:
    return f"""Perform earnings analysis for {source_file} using ONLY the context below.

ANALYTICAL FRAMEWORK — evaluate internally on 4 dimensions (DO NOT output scores):
1. Revenue Performance — growth strength, beat/miss vs expectations
2. Profitability — gross margin, operating margin, EPS growth
3. Earnings Quality — recurring vs one-time components, cash flow
4. Management Guidance — forward outlook, confidence signals, risk disclosures

Final score using balanced judgment:
- Strong across all dimensions → 8-9
- Mixed signals              → 5-7
- Weak fundamentals          → 2-4

Return ONLY valid JSON, no markdown, no extra text:
{{
  "agent": "Earnings Analyst",
  "source_file": "{source_file}",
  "key_points": [
    "Revenue insight max 25 words",
    "Profitability insight max 25 words",
    "Earnings quality insight max 25 words",
    "Guidance insight max 25 words"
  ],
  "score": <float 0.0-10.0>,
  "reasoning": "2-3 sentences justifying score using the four dimensions with specific figures from context"
}}

Context:
{context}"""


# ── RAG RETRIEVAL ──
def get_context(company: str) -> str:
    query = "revenue growth profitability margins earnings guidance financial performance"
    results = retrieve_filtered(query=query, company=company, data_type="earnings_q1", k=4)
    if not results:
        return "No financial data available."
    return "\n\n".join(f"[Chunk {i+1}]: {chunk.strip()}" for i, chunk in enumerate(results))


# ── SAFE JSON PARSE ──
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


# ── GROQ CALL (2-level: system + user) ──
def run_earnings_llm(company: str, context: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": EARNINGS_SYSTEM_PROMPT},
            {"role": "user",   "content": build_earnings_user_prompt(company, context)}
        ],
        temperature=0.3,
        max_tokens=1200
    )
    return response.choices[0].message.content.strip()


# ── MAIN AGENT ──
def earnings_agent(company: str) -> dict:
    context = get_context(company)
    raw = run_earnings_llm(company, context)
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    parsed = safe_parse(raw)
    if "score" in parsed:
        parsed["score"] = round(min(float(parsed["score"]), 10.0), 1)
    return parsed


def save_output(result: dict, company: str) -> None:
    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/{company.lower()}_earnings.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Saved → {filename}")


if __name__ == "__main__":
    initialize_rag()
    for company in ["Apple", "Tesla"]:
        result = earnings_agent(company)
        print(json.dumps(result, indent=2))
        save_output(result, company)