"""
cross_model_benchmark.py
========================
Person 6 — Task 2: Cross-Model Benchmark Runs

Runs 5 LLMs across 4 conditions on 15 sampled reports:
  Models: Gemini 2.5 Flash, Llama 4 Maverick (via OpenRouter),
          DeepSeek-R1 (via OpenRouter), Claude Sonnet 4 (free tier),
          GPT-4o → replaced with Gemini Pro (free) since GPT-4o is paid

  Conditions:
    1. zero_shot     — no RAG, direct generation
    2. standard_rag  — RAG with generic embedding
    3. multi_agent   — multi-agent without debate
    4. findebate     — full FinDebate pipeline (p5 outputs already done)

  For conditions 1-3, we SIMULATE by having the LLM generate a report
  from just the executive_summary + investment context extracted from
  the p5 JSON (since we don't have the raw transcripts in this module).

  The LLM-as-judge then scores each generated report on the 8 dimensions.

BATCHING: Use SLURM_ARRAY_TASK_ID to split across nodes.
  - Each array task handles one model
  - Within each model, iterate over files sequentially

FREE API SOURCES:
  - Gemini 2.5 Flash: Google AI Studio (free tier, 15 RPM)
  - Llama 4 Maverick: via OpenRouter free tier
  - DeepSeek-R1: via OpenRouter free tier
  - Claude Sonnet 4: via OpenRouter free tier (limited)
  - "GPT-4o equivalent": Gemini 1.5 Pro (free) as substitute
"""

import os
import json
import time
import logging
import argparse
import re
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cross_model_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
BENCHMARK_OUTPUT_DIR = os.getenv("BENCHMARK_OUTPUT_DIR", "outputs/cross_model")
JUDGE_RESULTS_DIR = os.getenv("JUDGE_RESULTS_DIR", "outputs/stats")

SLEEP_GEMINI = float(os.getenv("SLEEP_GEMINI", "5.0"))      # 15 RPM free tier
SLEEP_OPENROUTER = float(os.getenv("SLEEP_OPENROUTER", "3.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# The 15 sampled reports from the paper (10 ECTSum + 5 professional)
# These are the ones that exist in the outputs folder
SAMPLED_REPORTS_15 = [
    # 10 ECTSum subset — representative selection
    "ABM_q3_2021_p5_output.json",
    "AME_q1_2021_p5_output.json",
    "CMI_q1_2014_p5_output.json",
    "CMI_q3_2014_p5_output.json",
    "DE_q1_2014_p5_output.json",
    "DOV_q2_2020_p5_output.json",
    "GD_q1_2021_p5_output.json",
    "LH_q3_2021_p5_output.json",
    "MSI_q3_2021_p5_output.json",
    "NEE_q3_2021_p5_output.json",
    # 5 professional subset
    "DNB_q2_2021_p5_output.json",
    "FIS_q4_2020_p5_output.json",
    "GCO_q1_2022_p5_output.json",
    "HTH_q4_2020_p5_output.json",
    "TT_q1_2021_p5_output.json",
]

# Model definitions
MODELS = {
    "gemini_20_flash": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash",
        "display": "Gemini 2.0 Flash",
        "sleep": SLEEP_GEMINI,
    },
    "llama4_maverick": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-4-maverick:free",
        "display": "Llama 4 Maverick",
        "sleep": SLEEP_OPENROUTER,
    },
    "deepseek_r1": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1:free",
        "display": "DeepSeek-R1",
        "sleep": SLEEP_OPENROUTER,
    },
    "claude_sonnet4": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4-5:free",
        "display": "Claude Sonnet 4",
        "sleep": SLEEP_OPENROUTER,
    },
    "gpt4o_equiv": {
        "provider": "gemini",
        "model_id": "gemini-1.5-pro",
        "display": "GPT-4o (Gemini-1.5-Pro substitute — free)",
        "sleep": SLEEP_GEMINI,
    },
}

CONDITIONS = ["zero_shot", "standard_rag", "multi_agent", "findebate"]

# ── Prompt templates for each condition ───────────────────────────────────────

ZERO_SHOT_PROMPT = """You are a professional financial analyst.
Based on the following brief context about a company's earnings, write a concise institutional investment report.

Company: {company}
Available context: {context}

Write a structured financial analysis report with:
- Executive Summary (2-3 sentences)
- Key Financial Highlights
- Investment Recommendation (Long/Short/Neutral for 1-day, 1-week, 1-month)
- Risk Assessment

Keep it professional and fact-based. Target 400-600 words."""

STANDARD_RAG_PROMPT = """You are a professional financial analyst with access to retrieved earnings call excerpts.
Use the following retrieved context to write an investment analysis report.

Company: {company}
Retrieved Context (from RAG system): {context}

Write a structured financial analysis report with:
- Executive Summary
- Financial Performance Analysis (use the retrieved metrics)
- Market Outlook
- Investment Recommendation (Long/Short/Neutral for 1-day, 1-week, 1-month with conviction %)
- Risk Factors

Be specific, cite the retrieved data, target 500-700 words."""

MULTI_AGENT_PROMPT = """You are a report synthesis agent combining outputs from 5 specialized financial analysts.
The agents have analyzed: earnings performance, market prediction, sentiment, valuation, and risk.

Company: {company}
Multi-agent analysis inputs: {context}

Synthesize all perspectives into ONE unified investment report with:
- Overall Investment Stance (BULLISH/BEARISH/NEUTRAL)
- Executive Summary covering all 5 analytical dimensions
- Detailed Investment Recommendations (1-day, 1-week, 1-month positions with conviction %)
- Risk-Reward Analysis
- Investment Conclusion

Target 600-800 words. Be comprehensive and institutional-grade."""

CONDITION_PROMPTS = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "standard_rag": STANDARD_RAG_PROMPT,
    "multi_agent": MULTI_AGENT_PROMPT,
}

# Judge prompt (same 8 dimensions)
JUDGE_PROMPT = """You are an expert financial report evaluator.
Evaluate this report on 8 dimensions using a 4-point scale (1=poor, 4=excellent):

1. readability - Clarity and fluency of language
2. linguistic_abstractness - Degree of synthesis beyond raw data
3. coherence - Logical flow and structural clarity
4. financial_key_point_coverage - Coverage of earnings highlights (revenue, profit, margins, guidance)
5. background_context_adequacy - Historical/industry context provision
6. management_sentiment_conveyance - Accuracy of management tone reflection
7. future_outlook_analysis - Quality of guidance and forecast reporting
8. factual_accuracy - Alignment of statements with source data

Return ONLY valid JSON (no markdown, no preamble):
{{
  "readability": <1-4>,
  "linguistic_abstractness": <1-4>,
  "coherence": <1-4>,
  "financial_key_point_coverage": <1-4>,
  "background_context_adequacy": <1-4>,
  "management_sentiment_conveyance": <1-4>,
  "future_outlook_analysis": <1-4>,
  "factual_accuracy": <1-4>
}}

REPORT:
{report}"""

DIMENSIONS = [
    "readability", "linguistic_abstractness", "coherence",
    "financial_key_point_coverage", "background_context_adequacy",
    "management_sentiment_conveyance", "future_outlook_analysis", "factual_accuracy"
]


# ── API Callers ───────────────────────────────────────────────────────────────

def call_gemini(model_id: str, prompt: str) -> str:
    """Call Gemini via google-genai SDK."""
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model=model_id, contents=prompt)
    return response.text.strip()


def call_openrouter(model_id: str, prompt: str) -> str:
    """Call OpenRouter (free tier) via REST API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://findebate-research.local",
        "X-Title": "FinDebate Research"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.6,
        "top_p": 0.85
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_model(model_key: str, prompt: str) -> str:
    """Unified model caller with retry logic."""
    cfg = MODELS[model_key]
    provider = cfg["provider"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "gemini":
                return call_gemini(cfg["model_id"], prompt)
            elif provider == "openrouter":
                return call_openrouter(cfg["model_id"], prompt)
        except Exception as e:
            err = str(e)
            logger.warning(f"[{model_key}] Attempt {attempt} failed: {err[:200]}")
            if attempt < MAX_RETRIES:
                wait = cfg["sleep"] * (2 ** attempt)
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)

    return ""


# ── Data Extraction ───────────────────────────────────────────────────────────

def extract_context_from_p5(data: dict) -> tuple[str, str]:
    """Extract company name and context from p5 output for simulation."""
    source = data.get("source_file", "Unknown")
    company = source.split("_")[0] if "_" in source else source

    debate = data.get("debate_result", {})
    parts = []

    exec_sum = debate.get("executive_summary", "")
    if exec_sum:
        parts.append(f"Executive Summary: {exec_sum[:600]}")

    inv_recs = debate.get("investment_recommendations", {})
    if inv_recs:
        parts.append("Investment Recommendations:")
        for k, v in inv_recs.items():
            if isinstance(v, dict):
                parts.append(f"  {k}: {v.get('position','')} ({v.get('conviction','')}) — {v.get('rationale','')[:200]}")

    risk = debate.get("risk_reward", "")
    if risk:
        parts.append(f"Risk-Reward: {str(risk)[:400]}")

    conclusion = debate.get("investment_conclusion", "")
    if conclusion:
        parts.append(f"Conclusion: {str(conclusion)[:400]}")

    return company, "\n".join(parts)


def extract_findebate_report(data: dict) -> str:
    """Extract full findebate report text for judging."""
    debate = data.get("debate_result", {})
    parts = []

    stance = debate.get("overall_stance", "")
    conviction = debate.get("overall_conviction", "")
    if stance:
        parts.append(f"STANCE: {stance} | CONVICTION: {conviction}")

    exec_sum = debate.get("executive_summary", "")
    if exec_sum:
        parts.append(f"\nEXECUTIVE SUMMARY:\n{exec_sum}")

    inv_recs = debate.get("investment_recommendations", {})
    if inv_recs:
        parts.append("\nINVESTMENT RECOMMENDATIONS:")
        for k, v in inv_recs.items():
            if isinstance(v, dict):
                parts.append(f"  {k}: {v.get('position','')} ({v.get('conviction','')}) — {v.get('rationale','')[:400]}")

    trust = str(debate.get("trust_enhancements", ""))[:800]
    if trust:
        parts.append(f"\nTRUST ENHANCEMENTS:\n{trust}")

    skeptic = str(debate.get("skeptic_risk_additions", ""))[:800]
    if skeptic:
        parts.append(f"\nRISK ADDITIONS:\n{skeptic}")

    risk_reward = str(debate.get("risk_reward", ""))[:600]
    if risk_reward:
        parts.append(f"\nRISK-REWARD:\n{risk_reward}")

    conclusion = str(debate.get("investment_conclusion", ""))[:600]
    if conclusion:
        parts.append(f"\nCONCLUSION:\n{conclusion}")

    return "\n".join(parts)


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge_report(report_text: str) -> dict:
    """Score a generated report using Gemini as the judge."""
    prompt = JUDGE_PROMPT.format(report=report_text[:4000])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_gemini("gemini-2.0-flash", prompt)
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            raw = raw.strip()

            match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if match:
                raw = match.group()

            scores = json.loads(raw)
            for dim in DIMENSIONS:
                if dim not in scores:
                    scores[dim] = 2
                else:
                    scores[dim] = max(1, min(4, int(scores[dim])))
            return scores

        except Exception as e:
            logger.warning(f"Judge attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_GEMINI * attempt)

    return {dim: 0 for dim in DIMENSIONS}


# ── Main Benchmark Loop ───────────────────────────────────────────────────────

def load_existing_benchmark(csv_path: str) -> set:
    """Load (model, condition, file) tuples already processed."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(zip(df["model"], df["condition"], df["file"]))
    except Exception:
        return set()


def run_benchmark(model_key: str):
    """Run all 4 conditions on the 15 sampled reports for one model."""
    Path(BENCHMARK_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(JUDGE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(JUDGE_RESULTS_DIR, f"benchmark_{model_key}.csv")
    already_done = load_existing_benchmark(csv_path)

    cfg = MODELS[model_key]
    logger.info(f"Starting benchmark for: {cfg['display']}")
    logger.info(f"Files to process: {len(SAMPLED_REPORTS_15)} × {len(CONDITIONS)} conditions")

    results = []

    for file_name in SAMPLED_REPORTS_15:
        path = os.path.join(OUTPUT_FOLDER, file_name)

        if not os.path.exists(path):
            logger.warning(f"File not found: {path} — skipping")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        company, context = extract_context_from_p5(data)
        findebate_report = extract_findebate_report(data)

        for condition in CONDITIONS:
            key = (model_key, condition, file_name)
            if key in already_done:
                logger.info(f"  Skip (done): {model_key} | {condition} | {file_name}")
                continue

            logger.info(f"  Running: {model_key} | {condition} | {file_name}")

            if condition == "findebate":
                # Use pre-existing p5 FinDebate output — just judge it
                report_text = findebate_report
                generation_success = True
            else:
                # Generate report using the model under this condition
                prompt_tmpl = CONDITION_PROMPTS[condition]
                gen_prompt = prompt_tmpl.format(company=company, context=context[:2000])

                generated = call_model(model_key, gen_prompt)
                time.sleep(cfg["sleep"])

                if not generated:
                    logger.error(f"  Generation failed for {model_key}/{condition}/{file_name}")
                    generation_success = False
                    report_text = ""
                else:
                    generation_success = True
                    report_text = generated

                    # Save generated report
                    gen_dir = os.path.join(BENCHMARK_OUTPUT_DIR, model_key, condition)
                    Path(gen_dir).mkdir(parents=True, exist_ok=True)
                    gen_path = os.path.join(gen_dir, file_name.replace("_p5_output.json", "_generated.txt"))
                    with open(gen_path, "w") as gf:
                        gf.write(report_text)

            # Judge the report
            if report_text:
                logger.info(f"    Judging...")
                scores = judge_report(report_text)
                time.sleep(SLEEP_GEMINI)
            else:
                scores = {dim: 0 for dim in DIMENSIONS}

            # Compute aggregate scores
            valid = [d for d in DIMENSIONS if scores.get(d, 0) > 0]
            dim1 = ["readability", "linguistic_abstractness", "coherence"]
            dim2 = ["financial_key_point_coverage", "background_context_adequacy",
                    "management_sentiment_conveyance", "future_outlook_analysis", "factual_accuracy"]

            row = {
                "model": model_key,
                "model_display": cfg["display"],
                "condition": condition,
                "file": file_name,
                "generation_success": generation_success,
            }
            row.update(scores)
            row["avg_textual_quality"] = round(
                sum(scores.get(d, 0) for d in dim1 if scores.get(d, 0) > 0) /
                max(1, len([d for d in dim1 if scores.get(d, 0) > 0])), 3) if valid else 0
            row["avg_financial_professionalism"] = round(
                sum(scores.get(d, 0) for d in dim2 if scores.get(d, 0) > 0) /
                max(1, len([d for d in dim2 if scores.get(d, 0) > 0])), 3) if valid else 0
            row["avg_overall"] = round(
                sum(scores.get(d, 0) for d in DIMENSIONS if scores.get(d, 0) > 0) /
                max(1, len(valid)), 3) if valid else 0

            results.append(row)
            already_done.add(key)

            # Incremental save
            pd.DataFrame([row]).to_csv(
                csv_path,
                mode="a",
                header=not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0,
                index=False
            )

            logger.info(f"    → avg_overall={row['avg_overall']}")

    logger.info(f"Benchmark complete for {cfg['display']}: {len(results)} new rows saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Model Benchmark for FinDebate")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
        help="Which model to benchmark"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model keys"
    )
    args = parser.parse_args()

    if args.list_models:
        for k, v in MODELS.items():
            print(f"  {k}: {v['display']} ({v['provider']})")
    else:
        # Check prerequisites
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set in .env!")
            exit(1)

        cfg = MODELS[args.model]
        if cfg["provider"] == "openrouter" and not OPENROUTER_API_KEY:
            logger.error(f"OPENROUTER_API_KEY not set! Required for {cfg['display']}")
            exit(1)

        run_benchmark(args.model)
