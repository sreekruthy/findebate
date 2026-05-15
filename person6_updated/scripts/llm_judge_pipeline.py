"""
llm_judge_pipeline.py
=====================
Person 6 — Task 1: LLM-as-Judge Pipeline
Uses GPT-4o evaluation framework with all 8 scoring dimensions from Appendix E.
Since GPT-4o is paid, we use Gemini (free) as the judge LLM.

Dimensions (from paper Appendix E):
1. Readability
2. Linguistic Abstractness
3. Coherence
4. Financial Key Point Coverage
5. Background Context Adequacy
6. Management Sentiment Conveyance
7. Future Outlook Analysis
8. Factual Accuracy

Scale: 1 (poor) to 4 (excellent)
"""

import os
import json
import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("llm_judge_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
RESULTS_CSV = os.getenv("RESULTS_CSV", "outputs/llm_judge/llm_judge_results.csv")
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "4.0"))  # rate limit safety
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BATCH_START = int(os.getenv("BATCH_START", "0"))
BATCH_END = int(os.getenv("BATCH_END", "999"))

# 8 dimensions from Appendix E of the paper
DIMENSIONS = [
    "readability",
    "linguistic_abstractness",
    "coherence",
    "financial_key_point_coverage",
    "background_context_adequacy",
    "management_sentiment_conveyance",
    "future_outlook_analysis",
    "factual_accuracy"
]

DIMENSION_DEFINITIONS = {
    "readability": "Clarity and fluency of the report's language; grammar, style, and ease of reading.",
    "linguistic_abstractness": "Degree of summarization and synthesis beyond raw data repetition.",
    "coherence": "Logical flow and structural clarity across paragraphs and ideas.",
    "financial_key_point_coverage": "Inclusion of core earnings highlights (revenue, profit, margins, guidance).",
    "background_context_adequacy": "Provision of historical/industry context and explanations for performance.",
    "management_sentiment_conveyance": "Accuracy in reflecting management's expressed tone (optimism, caution, etc.).",
    "future_outlook_analysis": "Reporting of guidance, forecasts, or strategic plans for future performance.",
    "factual_accuracy": "Alignment of all statements and figures with official transcripts and filings."
}

JUDGE_PROMPT = """You are an expert financial report evaluator. 
Evaluate the following financial analysis report on exactly 8 dimensions using a 4-point scale:

SCORING SCALE:
1 = Poor / Not reported
2 = Reported but not useful / Below average
3 = Reported and reasonable / Good
4 = Reported and insightful / Excellent

DIMENSIONS TO EVALUATE:
1. readability - {readability}
2. linguistic_abstractness - {linguistic_abstractness}
3. coherence - {coherence}
4. financial_key_point_coverage - {financial_key_point_coverage}
5. background_context_adequacy - {background_context_adequacy}
6. management_sentiment_conveyance - {management_sentiment_conveyance}
7. future_outlook_analysis - {future_outlook_analysis}
8. factual_accuracy - {factual_accuracy}

Return ONLY valid JSON with exactly these keys (no markdown, no explanation):
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

REPORT TO EVALUATE:
{report}
"""


def build_client():
    """Build the Gemini client."""
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to build Gemini client: {e}")
        raise


def extract_report_text(data: dict) -> str:
    """Extract the final debate report text from a p5 output JSON."""
    debate = data.get("debate_result", {})

    # Build a readable report summary from the debate_result fields
    parts = []

    stance = debate.get("overall_stance", "")
    conviction = debate.get("overall_conviction", "")
    if stance:
        parts.append(f"OVERALL STANCE: {stance} (Conviction: {conviction})")

    exec_summary = debate.get("executive_summary", "")
    if exec_summary:
        parts.append(f"\nEXECUTIVE SUMMARY:\n{exec_summary}")

    inv_recs = debate.get("investment_recommendations", {})
    if inv_recs:
        parts.append("\nINVESTMENT RECOMMENDATIONS:")
        for horizon, rec in inv_recs.items():
            if isinstance(rec, dict):
                pos = rec.get("position", "")
                conv = rec.get("conviction", "")
                rat = rec.get("rationale", "")
                parts.append(f"  {horizon.upper()}: {pos} ({conv}) — {rat[:300]}")
            else:
                parts.append(f"  {horizon}: {str(rec)[:300]}")

    trust = debate.get("trust_enhancements", "")
    if trust:
        parts.append(f"\nTRUST ENHANCEMENTS (supporting evidence):\n{str(trust)[:800]}")

    skeptic = debate.get("skeptic_risk_additions", "")
    if skeptic:
        parts.append(f"\nSKEPTIC RISK ADDITIONS:\n{str(skeptic)[:800]}")

    risk_reward = debate.get("risk_reward", "")
    if risk_reward:
        parts.append(f"\nRISK-REWARD ANALYSIS:\n{str(risk_reward)[:800]}")

    conclusion = debate.get("investment_conclusion", "")
    if conclusion:
        parts.append(f"\nINVESTMENT CONCLUSION:\n{str(conclusion)[:600]}")

    return "\n".join(parts)


def call_judge_with_retry(client, prompt: str, file_name: str) -> dict:
    """Call the Gemini judge with retry logic."""
    import re
    from google import genai

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            text = response.text.strip()

            # Strip markdown fences
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            text = text.strip()

            # Try to find a JSON object if extra text present
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if match:
                text = match.group()

            scores = json.loads(text)

            # Validate all 8 dimensions are present
            for dim in DIMENSIONS:
                if dim not in scores:
                    logger.warning(f"{file_name}: Missing dimension '{dim}', setting to 2")
                    scores[dim] = 2
                else:
                    # Clamp to 1-4
                    scores[dim] = max(1, min(4, int(scores[dim])))

            return scores

        except json.JSONDecodeError as e:
            logger.warning(f"{file_name}: Attempt {attempt} JSON parse error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_CALLS * attempt)
        except Exception as e:
            err = str(e)
            logger.warning(f"{file_name}: Attempt {attempt} API error: {err}")
            if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                wait = SLEEP_BETWEEN_CALLS * (2 ** attempt)
                logger.info(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_CALLS)

    logger.error(f"{file_name}: All {MAX_RETRIES} attempts failed.")
    return {dim: 0 for dim in DIMENSIONS}  # 0 = failed marker


def load_existing_results(csv_path: str) -> set:
    """Load already-processed files to support resume."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df["file"].tolist())
    except Exception:
        return set()


def run_judge(args):
    """Main judge loop with batching support."""
    client = build_client()

    output_dir = Path(RESULTS_CSV).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    all_files = sorted([
        f for f in os.listdir(OUTPUT_FOLDER)
        if f.endswith(".json") and f != "final_summary.json"
    ])

    # Apply batch slicing (for SLURM array jobs)
    batch_files = all_files[BATCH_START:BATCH_END]
    logger.info(f"Processing files {BATCH_START} to {BATCH_END}: {len(batch_files)} files")

    # Resume support
    already_done = load_existing_results(RESULTS_CSV)
    logger.info(f"Already processed: {len(already_done)} files")

    prompt_template = JUDGE_PROMPT.format(**DIMENSION_DEFINITIONS, report="{report}")

    results = []

    for i, file_name in enumerate(batch_files):
        if file_name in already_done:
            logger.info(f"[{i+1}/{len(batch_files)}] Skipping (already done): {file_name}")
            continue

        path = os.path.join(OUTPUT_FOLDER, file_name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Cannot read {file_name}: {e}")
            continue

        if "debate_result" not in data:
            logger.warning(f"Skipping {file_name}: no 'debate_result' key")
            continue

        report_text = extract_report_text(data)
        # Trim to 3000 chars — sufficient to score all 8 dimensions without
        # wasting tokens on repetitive tail sections of long debate reports.
        report_text = report_text[:3000]
        prompt = JUDGE_PROMPT.format(**DIMENSION_DEFINITIONS, report=report_text)

        logger.info(f"[{i+1}/{len(batch_files)}] Judging: {file_name} (~{len(prompt)//4} tokens)")

        scores = call_judge_with_retry(client, prompt, file_name)
        scores["file"] = file_name
        scores["model"] = "findebate"
        scores["condition"] = "findebate"

        # Compute averages
        valid_dims = [d for d in DIMENSIONS if scores.get(d, 0) > 0]
        if valid_dims:
            dim1 = ["readability", "linguistic_abstractness", "coherence"]
            dim2 = ["financial_key_point_coverage", "background_context_adequacy",
                    "management_sentiment_conveyance", "future_outlook_analysis", "factual_accuracy"]
            scores["avg_textual_quality"] = round(
                sum(scores.get(d, 0) for d in dim1 if scores.get(d, 0) > 0) /
                max(1, len([d for d in dim1 if scores.get(d, 0) > 0])), 3)
            scores["avg_financial_professionalism"] = round(
                sum(scores.get(d, 0) for d in dim2 if scores.get(d, 0) > 0) /
                max(1, len([d for d in dim2 if scores.get(d, 0) > 0])), 3)
            scores["avg_overall"] = round(
                sum(scores.get(d, 0) for d in DIMENSIONS if scores.get(d, 0) > 0) /
                max(1, len([d for d in DIMENSIONS if scores.get(d, 0) > 0])), 3)
        else:
            scores["avg_textual_quality"] = 0
            scores["avg_financial_professionalism"] = 0
            scores["avg_overall"] = 0

        results.append(scores)

        # Append to CSV incrementally (safe for SLURM)
        pd.DataFrame([scores]).to_csv(
            RESULTS_CSV,
            mode="a",
            header=not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0,
            index=False
        )

        logger.info(f"  → avg_overall={scores['avg_overall']}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    logger.info(f"Batch complete. Processed {len(results)} new files.")
    return results


def summarize_results():
    """Print a summary table of all judge results."""
    if not os.path.exists(RESULTS_CSV):
        logger.warning("No results CSV found yet.")
        return

    df = pd.read_csv(RESULTS_CSV)
    df = df[df["avg_overall"] > 0]  # exclude failed

    logger.info(f"\n{'='*60}")
    logger.info(f"LLM JUDGE SUMMARY ({len(df)} reports evaluated)")
    logger.info(f"{'='*60}")

    for dim in DIMENSIONS + ["avg_textual_quality", "avg_financial_professionalism", "avg_overall"]:
        if dim in df.columns:
            logger.info(f"  {dim:40s}: {df[dim].mean():.3f} ± {df[dim].std():.3f}")

    summary_path = RESULTS_CSV.replace(".csv", "_summary.json")
    summary = {
        dim: {"mean": round(df[dim].mean(), 4), "std": round(df[dim].std(), 4)}
        for dim in DIMENSIONS + ["avg_textual_quality", "avg_financial_professionalism", "avg_overall"]
        if dim in df.columns
    }
    summary["n_reports"] = len(df)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge Pipeline for FinDebate")
    parser.add_argument("--summarize", action="store_true", help="Just print summary of existing results")
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER, help="Folder with p5 output JSONs")
    parser.add_argument("--batch-start", type=int, default=BATCH_START)
    parser.add_argument("--batch-end", type=int, default=BATCH_END)
    args = parser.parse_args()

    if args.output_folder:
        OUTPUT_FOLDER = args.output_folder
    if args.batch_start is not None:
        BATCH_START = args.batch_start
    if args.batch_end is not None:
        BATCH_END = args.batch_end

    if args.summarize:
        summarize_results()
    else:
        run_judge(args)
        summarize_results()
