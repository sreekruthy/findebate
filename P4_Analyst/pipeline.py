import os
import json
import time
import chromadb
import argparse
from datetime import datetime
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Config ───────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CHROMA_PATH    = os.path.expanduser("~/findebate/findebate_chromadb")  # FIXED
OUTPUT_DIR     = os.path.expanduser("~/findebate/outputs")
GITHUB_REPO    = "sreekruthy/findebate"
GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN", "")
GITHUB_BRANCH  = "Person4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── ChromaDB Setup ───────────────────────────────────────────
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_collection("findebate_rag")
embed_model   = SentenceTransformer("FinLang/finance-embeddings-investopedia")
print(f"Connected. Total chunks: {collection.count()}")

# ── Gemini Setup ─────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

CALLS_PER_MINUTE = 8
call_timestamps  = []

def wait_for_rate_limit():
    global call_timestamps
    now = time.time()
    call_timestamps = [t for t in call_timestamps if now - t < 60]
    if len(call_timestamps) >= CALLS_PER_MINUTE:
        oldest    = call_timestamps[0]
        wait_time = 60 - (now - oldest) + 2
        print(f"  Rate limit reached. Waiting {wait_time:.0f}s...")
        time.sleep(wait_time)
        now = time.time()
        call_timestamps = [t for t in call_timestamps if now - t < 60]
    call_timestamps.append(time.time())

def call_gemini(system_prompt, user_prompt, retries=3, max_tokens=3000):
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    for attempt in range(retries):
        wait_for_rate_limit()
        print(f"  Attempt {attempt+1}/{retries}...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    top_p=0.85,
                    max_output_tokens=max_tokens,
                    http_options=types.HttpOptions(timeout=120000)
                )
            )
            print(f"  Success!")
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                print(f"  Timeout. Waiting 15s...")
                time.sleep(15)
            elif "429" in error_str or "quota" in error_str or "rate" in error_str:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                time.sleep(10)
    print("  All attempts failed.")
    return None

# ── RAG Functions ────────────────────────────────────────────
DIMENSION_QUERIES = {
    "general_financial": [
        "financial performance revenue earnings beat miss surprise results",
        "guidance outlook forecast expectations future performance strategy",
        "growth trends margin expansion profitability cash flow competitive"
    ],
    "specialized_metrics": [
        "net interest margin NIM loan deposits credit quality asset quality",
        "non-performing assets NPAs charge-offs provision loan losses",
        "return on assets ROA return on equity ROE efficiency ratio capital adequacy"
    ],
    "market_sentiment_risk": [
        "management confidence sentiment optimistic cautious positive negative tone",
        "risks challenges concerns headwinds uncertainties market conditions",
        "credit risk operational risk market risk liquidity risk hedging"
    ],
    "multi_query_integration": [
        "short-term immediate near-term weekly monthly quarterly timeline",
        "comparative performance benchmarking cross-functional analysis",
        "comprehensive integrated multi-dimensional longitudinal tracking"
    ]
}

AGENT_DIMENSION_MAP = {
    "valuation_agent" : ["specialized_metrics", "general_financial"],
    "risk_agent"      : ["market_sentiment_risk", "specialized_metrics"]
}

def retrieve(query, top_k=5):
    q_emb   = embed_model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "chunk"       : doc,
            "source_file" : meta["source_file"],
            "chunk_id"    : meta["chunk_index"],
            "score"       : round(1 - dist, 4)
        })
    return output

def retrieve_by_dimension(dimension, top_k=5):
    seen, merged = set(), []
    for query in DIMENSION_QUERIES[dimension]:
        for r in retrieve(query, top_k=top_k):
            uid = f"{r['source_file']}_chunk_{r['chunk_id']}"
            if uid not in seen:
                seen.add(uid)
                merged.append(r)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]

def get_agent_context(agent_name, source_file=None, top_k=5):
    dimensions = AGENT_DIMENSION_MAP[agent_name]
    seen, chunks = set(), []
    for dim in dimensions:
        for r in retrieve_by_dimension(dim, top_k=top_k*20):
            if source_file and r["source_file"] != source_file:
                continue
            uid = f"{r['source_file']}_chunk_{r['chunk_id']}"
            if uid not in seen:
                seen.add(uid)
                chunks.append(r)
    chunks.sort(key=lambda x: x["score"], reverse=True)
    # If no chunks found for source_file, fall back to general retrieval
    #but now we are skipping the files instead of making them hallicinate
    if not chunks:
        print(f"  ERROR: No chunks found for {source_file}, skipping")
        return []
    return chunks[:top_k]

def chunks_to_text(chunks):
    return "\n\n".join([
        f"[Source: {r['source_file']} | Score: {r['score']}]\n{r['chunk']}"
        for r in chunks
    ])

# ── Agents ───────────────────────────────────────────────────
def run_valuation_agent(source_file=None, top_k=5):
    print(f"  Running Valuation Agent on {source_file}...")
    chunks       = get_agent_context("valuation_agent", source_file=source_file, top_k=top_k)
    context_text = chunks_to_text(chunks)

    system_prompt = """You are a CFA charterholder and senior equity research analyst with 18+ years 
of experience building institutional-grade valuation assessments for major investment banks.

STANDARDS:
- Base ALL assessments on verifiable business fundamentals from the earnings call
- Maintain realistic confidence levels between 70-80%
- Apply sector-specific DCF considerations
- Use dynamic weight allocation across valuation methods"""

    user_prompt = f"""CRITICAL: Return ONLY valid JSON. No markdown. No explanation. No text outside JSON.

Based on the following earnings call excerpts, produce an institutional-grade valuation analysis.

EARNINGS CALL EXCERPTS:
{context_text}

Return ONLY this exact JSON structure:
{{
    "agent": "Valuation Analyst",
    "source_file": "{source_file or 'all'}",
    "timestamp": "{datetime.now().isoformat()}",
    "investment_stance": "UNDERVALUED or FAIRLY VALUED or OVERVALUED",
    "conviction_level": "75%",
    "key_points": [
        "Key valuation insight 1",
        "Key valuation insight 2",
        "Key valuation insight 3",
        "Key valuation insight 4",
        "Key valuation insight 5"
    ],
    "dcf_signals": {{
        "growth_outlook": "brief growth assessment",
        "margin_trajectory": "brief margin assessment",
        "capital_efficiency": "brief capital allocation assessment"
    }},
    "scenarios": {{
        "bull_case": "brief bull case",
        "base_case": "brief base case",
        "bear_case": "brief bear case"
    }},
    "score": <number between 0-10 based on your analysis where 0-3 is weak, 4-6 is mixed, 7-8 is strong, 9-10 is exceptional>,
    "reasoning": "Concise paragraph justifying stance and score"
}}"""

    raw = call_gemini(system_prompt, user_prompt, max_tokens=5000)
    if not raw:
        return None
    try:
        clean = raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        result = json.loads(clean)
        print(f"  Valuation done: {result['investment_stance']} | Score: {result['score']}")
        return result
    except Exception as e:
        print(f"  JSON error: {e}")
        return {"agent": "Valuation Analyst", "raw": raw}


def run_risk_agent(source_file=None, top_k=5):
    print(f"  Running Risk Agent on {source_file}...")
    chunks       = get_agent_context("risk_agent", source_file=source_file, top_k=top_k)
    context_text = chunks_to_text(chunks)

    system_prompt = """You are a senior risk management specialist with extensive experience 
in equity risk assessment for major asset management firms.

STANDARDS:
- Provide BALANCED risk assessment
- Evaluate credit risk, interest rate risk, liquidity risk
- Maintain realistic confidence levels between 70-80%
- Deliver actionable position sizing guidance"""

    user_prompt = f"""CRITICAL: Return ONLY valid JSON. No markdown. No explanation. No text outside JSON.

Based on the following earnings call excerpts, produce an institutional-grade risk assessment.

EARNINGS CALL EXCERPTS:
{context_text}

Return ONLY this exact JSON structure:
{{
    "agent": "Risk Analyst",
    "source_file": "{source_file or 'all'}",
    "timestamp": "{datetime.now().isoformat()}",
    "overall_risk_rating": "LOW or MODERATE or HIGH or VERY HIGH",
    "conviction_level": "75%",
    "key_points": [
        "Key risk insight 1",
        "Key risk insight 2",
        "Key risk insight 3",
        "Key risk insight 4",
        "Key risk insight 5"
    ],
    "risk_dimensions": {{
        "credit_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}},
        "interest_rate_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}},
        "liquidity_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}},
        "operational_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}}
    }},
    "position_sizing": {{
        "recommended_position": "X% of portfolio",
        "max_position": "X% of portfolio",
        "hedge_strategies": ["strategy 1", "strategy 2"]
    }},
    "risk_triggers": ["Trigger 1", "Trigger 2", "Trigger 3"],
    "score": <number between 0-10 based on risk level where 0-3 is very high risk, 4-6 is moderate, 7-8 is low risk, 9-10 is minimal risk>,
    "reasoning": "Concise paragraph justifying overall risk rating"
}}"""

    raw = call_gemini(system_prompt, user_prompt, max_tokens=5000)
    if not raw:
        return None
    try:
        clean = raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        result = json.loads(clean)
        print(f"  Risk done: {result['overall_risk_rating']} | Score: {result['score']}")
        return result
    except Exception as e:
        print(f"  JSON error: {e}")
        return {"agent": "Risk Analyst", "raw": raw}


def run_report_synthesizer(agent_outputs, source_file=None):
    print(f"  Running Report Synthesizer on {source_file}...")
    agents_summary = json.dumps(agent_outputs, indent=2)

    system_prompt = """You are a Managing Director at a top-tier investment bank.
Portfolio managers will make Long/Short decisions for 1-day, 1-week, 1-month 
timeframes based on your analysis.

STANDARDS:
- Synthesize ALL analyst perspectives into one coherent narrative
- Maintain realistic conviction levels between 70-80%
- Never contradict yourself across timeframes"""

    user_prompt = f"""CRITICAL: Return ONLY valid JSON. No markdown. No explanation. No text outside JSON.
The JSON must contain EXACTLY these keys: agent, source_file, timestamp, overall_stance,
overall_conviction, executive_summary, multi_dimensional_synthesis, investment_recommendations,
risk_reward, investment_conclusion, agent_scores_summary, reasoning.
The investment_recommendations key must contain EXACTLY: one_day, one_week, one_month.
Each of those must contain EXACTLY: position, conviction, expected_direction.

Synthesize these analyst outputs into a single institutional investment report.

ANALYST OUTPUTS:
{agents_summary}

IMPORTANT: Do NOT default to NEUTRAL. Make a decisive BULLISH or BEARISH call based on the evidence. Only use NEUTRAL if evidence is truly mixed and contradictory.

Return ONLY this exact JSON structure:
{{
    "agent": "Report Synthesizer",
    "source_file": "{source_file or 'all'}",
    "timestamp": "{datetime.now().isoformat()}",
    "overall_stance": "BULLISH or NEUTRAL or BEARISH",
    "overall_conviction": "75%",
    "executive_summary": "150 word summary",
    "multi_dimensional_synthesis": {{
        "earnings_highlights": "key earnings insights",
        "market_positioning": "market dynamics summary",
        "management_sentiment": "sentiment assessment",
        "valuation_summary": "valuation conclusion",
        "risk_profile": "overall risk summary"
    }},
    "investment_recommendations": {{
        "one_day": {{
            "position": "LONG or SHORT or NEUTRAL",
            "conviction": "75%",
            "expected_direction": "brief direction"
        }},
        "one_week": {{
            "position": "LONG or SHORT or NEUTRAL",
            "conviction": "75%",
            "expected_direction": "brief direction"
        }},
        "one_month": {{
            "position": "LONG or SHORT or NEUTRAL",
            "conviction": "75%",
            "expected_direction": "brief direction"
        }}
    }},
    "risk_reward": {{
        "upside_catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"],
        "downside_risks": ["risk 1", "risk 2", "risk 3"],
        "position_sizing": "X% of portfolio",
        "hedge_strategies": ["hedge 1", "hedge 2"]
    }},
    "investment_conclusion": {{
        "final_stance": "BULLISH or NEUTRAL or BEARISH",
        "conviction": "75%",
        "top_3_insights": [
            "Actionable insight 1",
            "Actionable insight 2",
            "Actionable insight 3"
        ]
    }},
    "agent_scores_summary": {{
        "valuation_score": <copy the score from valuation agent output>,
        "risk_score": <copy the score from risk agent output>,
        "composite_score": <average of valuation and risk scores>
    }},
    "reasoning": "Concise paragraph explaining synthesis"
}}"""

    raw = call_gemini(system_prompt, user_prompt, max_tokens=8000)
    if not raw:
        return None
    try:
        clean = raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        result = json.loads(clean)
        print(f"  Synthesis done: {result['overall_stance']}")
        return result
    except Exception as e:
        print(f"  JSON error: {e}")
        return {"agent": "Report Synthesizer", "raw": raw}


# ── GitHub Push ──────────────────────────────────────────────
def push_to_github(filename, content, repo, branch, token):
    import base64
    import requests

    path    = f"outputs/{os.path.basename(filename)}"
    url     = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept"       : "application/vnd.github.v3+json"
    }

    sha = None
    check = requests.get(url, headers=headers)
    if check.status_code == 200:
        sha = check.json()["sha"]

    encoded = base64.b64encode(
        json.dumps(content, indent=2).encode()
    ).decode()

    payload = {
        "message" : f"Add output: {os.path.basename(filename)}",
        "content" : encoded,
        "branch"  : branch
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        print(f"  Pushed to GitHub: {path}")
        return True
    else:
        print(f"  GitHub push failed: {response.status_code}")
        return False


# ── Full Pipeline ────────────────────────────────────────────
def run_full_pipeline(source_file, top_k=5):
    print(f"\n{'='*60}")
    print(f"Processing: {source_file}")
    print(f"{'='*60}")

    try:
        valuation = run_valuation_agent(source_file=source_file, top_k=top_k)
        risk      = run_risk_agent(source_file=source_file, top_k=top_k)

        valid_outputs = [o for o in [valuation, risk] if o is not None]
        synthesis     = run_report_synthesizer(
            agent_outputs=valid_outputs,
            source_file=source_file
        )

        if valuation and risk and synthesis:
            output = {
                "source_file" : source_file,
                "timestamp"   : datetime.now().isoformat(),
                "agents"      : {
                    "valuation" : valuation,
                    "risk"      : risk,
                    "synthesis" : synthesis
                }
            }

            # Save locally
            local_path = f"{OUTPUT_DIR}/{source_file}_p4_output.json"
            with open(local_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  Saved: {local_path}")

            # Push to GitHub
            if GITHUB_TOKEN:
                push_to_github(
                    local_path, output,
                    GITHUB_REPO, GITHUB_BRANCH,
                    GITHUB_TOKEN
                )

            # FIXED — safe key access with .get()
            recommendations = synthesis.get("investment_recommendations", {})
            one_day         = recommendations.get("one_day", {})
            one_week        = recommendations.get("one_week", {})
            one_month       = recommendations.get("one_month", {})

            return {
                "source_file" : source_file,
                "status"      : "success",
                "stance"      : synthesis.get("overall_stance", "UNKNOWN"),
                "1day"        : one_day.get("position", "UNKNOWN"),
                "1week"       : one_week.get("position", "UNKNOWN"),
                "1month"      : one_month.get("position", "UNKNOWN"),
            }
        else:
            return {"source_file": source_file, "status": "partial_failure"}

    except Exception as e:
        print(f"Error on {source_file}: {e}")
        return {"source_file": source_file, "status": "failed", "error": str(e)}


# ── Batch Runner ─────────────────────────────────────────────
def get_all_source_files():
    results = collection.query(
        query_embeddings=embed_model.encode(
            ["financial earnings revenue"],
            convert_to_numpy=True
        ).tolist(),
        n_results=500,
        include=["metadatas"]
    )
    source_files = list(set([
        meta["source_file"]
        for meta in results["metadatas"][0]
    ]))
    source_files.sort()
    return source_files


def run_batch(file_list, top_k=5, start_from=0, end_at=None):
    total        = len(file_list)
    end_at       = end_at if end_at else total
    files_to_run = file_list[start_from:end_at]
    results      = []
    failed       = []

    print(f"Starting batch: {len(files_to_run)} transcripts")
    print(f"Resuming from index: {start_from}")

    for i, source_file in enumerate(files_to_run):
        idx    = start_from + i
        print(f"\nProgress: {idx+1}/{total}")
        result = run_full_pipeline(source_file, top_k=top_k)
        results.append(result)

        if result["status"] != "success":
            failed.append(source_file)

        # Checkpoint every 5 files
        if (i + 1) % 5 == 0:
            checkpoint = {
                "timestamp" : datetime.now().isoformat(),
                "completed" : idx + 1,
                "total"     : total,
                "results"   : results,
                "failed"    : failed
            }
            with open(f"{OUTPUT_DIR}/checkpoint.json", "w") as f:
                json.dump(checkpoint, f, indent=2)
            print(f"Checkpoint saved at {idx+1}/{total}")

    summary = {
        "timestamp"   : datetime.now().isoformat(),
        "total"       : total,
        "successful"  : len([r for r in results if r["status"] == "success"]),
        "failed"      : len(failed),
        "failed_files": failed,
        "results"     : results
    }

    with open(f"{OUTPUT_DIR}/batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"Successful: {summary['successful']}/{total}")
    print(f"Failed    : {summary['failed']}/{total}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="batch", help="batch or single")
    parser.add_argument("--file",       default=None,    help="single file name")
    parser.add_argument("--top_k",      default=5,       type=int)
    parser.add_argument("--start_from", default=0,       type=int)
    parser.add_argument("--end_at", default=None, type=int)
    args = parser.parse_args()

    if args.mode == "single":
        result = run_full_pipeline(args.file, top_k=args.top_k)
        print(json.dumps(result, indent=2))
    else:
        all_files = get_all_source_files()
        print(f"Found {len(all_files)} transcripts")
        run_batch(all_files, top_k=args.top_k, start_from=args.start_from, end_at=args.end_at)
