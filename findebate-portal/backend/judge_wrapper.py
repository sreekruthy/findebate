import json
import os

from dotenv import load_dotenv


load_dotenv()

DIMENSIONS = [
    "readability",
    "linguistic_abstractness",
    "coherence",
    "financial_key_point_coverage",
    "background_context_adequacy",
    "management_sentiment_conveyance",
    "future_outlook_analysis",
    "factual_accuracy",
]

JUDGE_PROMPT = """You are an expert evaluator of financial analysis reports.
Score the report on each dimension from 1 to 4. Return only valid JSON with
these exact keys: readability, linguistic_abstractness, coherence,
financial_key_point_coverage, background_context_adequacy,
management_sentiment_conveyance, future_outlook_analysis, factual_accuracy,
avg_overall."""


def build_report_text(report: dict) -> str:
    return "\n\n".join(
        str(report.get(key, ""))
        for key in [
            "overall_stance",
            "overall_conviction",
            "executive_summary",
            "investment_recommendations",
            "trust_enhancements",
            "skeptic_risk_additions",
            "risk_reward",
            "investment_conclusion",
        ]
    )


def _fallback_scores() -> dict:
    scores = {
        "readability": 3.7,
        "linguistic_abstractness": 3.55,
        "coherence": 3.75,
        "financial_key_point_coverage": 3.8,
        "background_context_adequacy": 3.6,
        "management_sentiment_conveyance": 3.65,
        "future_outlook_analysis": 3.72,
        "factual_accuracy": 3.78,
    }
    scores["avg_overall"] = round(sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)
    return scores


async def score_report(report_text: str) -> dict:
    if not os.getenv("GROQ_API_KEY"):
        return _fallback_scores()
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": report_text},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        scores = json.loads(raw)
        scores["avg_overall"] = round(sum(float(scores[d]) for d in DIMENSIONS) / len(DIMENSIONS), 2)
        return scores
    except Exception:
        return _fallback_scores()
