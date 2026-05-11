# sentiment_agent.py — Person 3 | Group A | Module 2
# Updated with confidence clamp (70-80%), source_file, behavioral_flags

import json
import os
from groq import Groq
from dotenv import load_dotenv
from sentiment_analyst.prompts import SENTIMENT_SYSTEM_PROMPT, build_sentiment_user_prompt

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _error_result(company: str, error_msg: str) -> dict:
    return {
        "agent": "Sentiment Analyst",
        "company": company,
        "source_file": "clean_clean_clean_earnings_q1.txt / clean_news1.txt",
        "key_points": ["Analysis failed — see reasoning for details"],
        "score": 5.0,
        "confidence": 0.70,
        "sentiment_label": "Neutral",
        "management_tone": "Neutral",
        "behavioral_flags": {
            "anchoring_detected": False,
            "overconfidence_detected": False,
            "loss_aversion_framing": "Not identified"
        },
        "reasoning": f"Error during analysis: {error_msg}",
    }


def run_sentiment_analyst(company: str, context_chunks: list) -> dict:
    if not context_chunks:
        return _error_result(company, "No context chunks provided.")

    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    user_prompt = build_sentiment_user_prompt(company, context)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1200,
        )

        raw_text = response.choices[0].message.content.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        result = json.loads(raw_text)

        # Clamp score 0-10
        result["score"] = max(0.0, min(10.0, float(result["score"])))

        # Enforce 70-80% confidence calibration requirement
        result["confidence"] = round(max(0.70, min(0.80, float(result.get("confidence", 0.75)))), 2)

        # Add source_file if missing
        result.setdefault("source_file", "clean_clean_clean_earnings_q1.txt / clean_news1.txt")

        # Add behavioral_flags if missing
        result.setdefault("behavioral_flags", {
            "anchoring_detected": False,
            "overconfidence_detected": False,
            "loss_aversion_framing": "Not identified"
        })

        return result

    except Exception as e:
        return _error_result(company, str(e))


if __name__ == "__main__":
    sample_chunks = [
        "Apple shares rose 3% after strong services revenue.",
        "CEO Tim Cook expressed confidence in Apple Intelligence driving upgrades.",
        "China sales declined 11.1% during the quarter.",
    ]
    result = run_sentiment_analyst("Apple", sample_chunks)
    print(json.dumps(result, indent=2))