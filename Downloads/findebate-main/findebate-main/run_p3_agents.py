"""
run_p3_agents.py — Person 3 | Group A | Module 2
=================================================
Runs Earnings, Market, Sentiment agents on all 64 ECTSum companies
using the ACTUAL agent implementations:
  - earnings_agent.py   (Earnings_Analyst/)
  - market_agent.py     (Market_Analyst/)
  - sentiment_analyst/prompts.py + sentiment_agent.py

This means the 2-level prompt structure, behavioral finance theories,
and 70-80% confidence calibration from your agent files are actually
used to generate the outputs — not inline prompts.

SETUP:
1. Place this file in findebate-main/findebate-main/
2. findebate_chromadb/ must contain chroma.sqlite3 + two UUID folders
3. .env must have GROQ_API_KEY
4. Run: python run_p3_agents.py

OUTPUT: p3_outputs/ABM_q3_2021_p3_output.json ... (64 files)
"""

import os
import sys
import json
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Add agent folders to path (same pattern as debate_engine.py) ──────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "Earnings_Analyst"))
sys.path.insert(0, os.path.join(BASE_DIR, "Market_Analyst"))
sys.path.insert(0, BASE_DIR)  # for sentiment_analyst package

# ── Import actual prompt builders from your agent files ───────────────────────
from earnings_agent import EARNINGS_SYSTEM_PROMPT, build_earnings_user_prompt
from market_agent import MARKET_SYSTEM_PROMPT, build_market_user_prompt
from sentiment_analyst.prompts import SENTIMENT_SYSTEM_PROMPT, build_sentiment_user_prompt

# ── Groq client ───────────────────────────────────────────────────────────────
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.path.join(BASE_DIR, "findebate_chromadb", "chroma.sqlite3")
OUTPUT_DIR     = os.path.join(BASE_DIR, "p3_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── ChromaDB retrieval via direct SQLite ──────────────────────────────────────
def get_context(source_file: str, max_chunks: int = 8) -> str:
    conn = sqlite3.connect(CHROMA_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT em.string_value
        FROM embeddings e
        JOIN embedding_metadata em  ON e.id = em.id AND em.key = 'chroma:document'
        JOIN embedding_metadata em2 ON e.id = em2.id AND em2.key = 'source_file'
        WHERE em2.string_value = ?
        LIMIT ?
    """, (source_file, max_chunks))
    rows = cursor.fetchall()
    conn.close()
    return "\n\n".join(r[0] for r in rows if r[0].strip())


def get_all_source_files():
    conn = sqlite3.connect(CHROMA_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT string_value FROM embedding_metadata
        WHERE key = 'source_file' ORDER BY string_value
    """)
    files = [r[0] for r in cursor.fetchall()]
    conn.close()
    return files


# ── Safe JSON parse ───────────────────────────────────────────────────────────
def safe_parse(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end   = text.rfind("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return {"error": "parse_failed", "raw": text[:300]}


# ── Earnings Agent — uses EARNINGS_SYSTEM_PROMPT + build_earnings_user_prompt ─
def run_earnings(source_file: str, context: str) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": EARNINGS_SYSTEM_PROMPT},
                {"role": "user",   "content": build_earnings_user_prompt(source_file, context)}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = safe_parse(raw)
        parsed["score"] = round(min(float(parsed.get("score", 5.0)), 10.0), 1)
        return parsed
    except Exception as e:
        return {
            "agent": "Earnings Analyst",
            "source_file": source_file,
            "key_points": ["Analysis failed — see reasoning"],
            "score": 5.0,
            "reasoning": f"Error: {str(e)}"
        }


# ── Market Agent — uses MARKET_SYSTEM_PROMPT + build_market_user_prompt ───────
def run_market(source_file: str, context: str) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": MARKET_SYSTEM_PROMPT},
                {"role": "user",   "content": build_market_user_prompt(source_file, context)}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = safe_parse(raw)
        parsed["score"] = round(min(float(parsed.get("score", 5.0)), 10.0), 1)
        return parsed
    except Exception as e:
        return {
            "agent": "Market Analyst",
            "source_file": source_file,
            "key_points": ["Analysis failed — see reasoning"],
            "score": 5.0,
            "reasoning": f"Error: {str(e)}"
        }


# ── Sentiment Agent — uses SENTIMENT_SYSTEM_PROMPT + build_sentiment_user_prompt
def run_sentiment(source_file: str, context: str) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                {"role": "user",   "content": build_sentiment_user_prompt(source_file, context)}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = safe_parse(raw)
        parsed["score"]      = round(max(0.0, min(10.0, float(parsed.get("score", 5.0)))), 1)
        parsed["confidence"] = round(max(0.70, min(0.80, float(parsed.get("confidence", 0.75)))), 2)
        parsed.setdefault("source_file", source_file)
        parsed.setdefault("behavioral_flags", {
            "anchoring_detected": False,
            "overconfidence_detected": False,
            "loss_aversion_framing": "Not identified"
        })
        return parsed
    except Exception as e:
        return {
            "agent": "Sentiment Analyst",
            "source_file": source_file,
            "key_points": ["Analysis failed — see reasoning"],
            "score": 5.0, "confidence": 0.70,
            "sentiment_label": "Neutral", "management_tone": "Neutral",
            "behavioral_flags": {
                "anchoring_detected": False,
                "overconfidence_detected": False,
                "loss_aversion_framing": "Not identified"
            },
            "reasoning": f"Error: {str(e)}"
        }


# ── Run one company through all 3 agents ─────────────────────────────────────
def run_company(source_file: str) -> dict:
    context = get_context(source_file, max_chunks=8)
    if not context.strip():
        return {"error": f"No context for {source_file}"}

    return {
        "source_file": source_file,
        "timestamp":   datetime.now().isoformat(),
        "agents": {
            "earnings":  run_earnings(source_file, context),
            "market":    run_market(source_file, context),
            "sentiment": run_sentiment(source_file, context)
        }
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    source_files = get_all_source_files()
    print(f"Found {len(source_files)} companies in ChromaDB")
    print(f"Using prompts from: earnings_agent.py, market_agent.py, sentiment_analyst/prompts.py")
    print(f"Output → {OUTPUT_DIR}\n")

    for i, sf in enumerate(source_files):
        out_path = os.path.join(OUTPUT_DIR, f"{sf}_p3_output.json")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(source_files)}] SKIP {sf} (already done)")
            continue

        print(f"[{i+1}/{len(source_files)}] Running {sf}...", end=" ", flush=True)
        result = run_company(sf)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        e = result["agents"].get("earnings",  {}).get("score", "?")
        m = result["agents"].get("market",    {}).get("score", "?")
        s = result["agents"].get("sentiment", {}).get("score", "?")
        print(f"E={e} M={m} S={s} ✓")

        time.sleep(8)

    print(f"\n✅ Done! {len(source_files)} files saved to {OUTPUT_DIR}")