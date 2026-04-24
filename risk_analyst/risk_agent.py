import json
import os
from groq import Groq
from dotenv import load_dotenv
from risk_analyst.prompts import RISK_SYSTEM_PROMPT, build_risk_user_prompt

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _error_result(company: str, error_msg: str) -> dict:
    return {
        "agent": "Risk Analyst",
        "company": company,
        "key_points": ["Analysis failed — see reasoning for details"],
        "score": 5.0,
        "risk_level": "Moderate",
        "identified_risks": [],
        "mitigation_factors": [],
        "reasoning": f"Error during analysis: {error_msg}",
    }


def run_risk_analyst(
    company: str,
    context_chunks: list,
) -> dict:
    if not context_chunks:
        return _error_result(company, "No context chunks provided.")

    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    user_prompt = build_risk_user_prompt(company, context)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": RISK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
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
        result["score"] = max(0.0, min(10.0, float(result["score"])))

        if not isinstance(result["identified_risks"], list):
            result["identified_risks"] = []

        return result

    except Exception as e:
        return _error_result(company, str(e))


if __name__ == "__main__":
    sample_chunks = [
        "Tesla automotive revenue plunged 20% from a year earlier.",
        "Net income plummeted 71% to $409 million.",
        "Tariffs will add approximately $900 million to costs.",
        "Energy storage revenue jumped 67% to $2.73 billion.",
    ]
    result = run_risk_analyst("Tesla", sample_chunks)
    print(json.dumps(result, indent=2))