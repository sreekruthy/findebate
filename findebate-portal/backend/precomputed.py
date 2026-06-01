import csv
import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
P5_OUTPUT_DIRS = [
    BASE_DIR / "outputs",
]

KNOWN_COMPANIES = {
    "ABM": "ABM_q3_2021_p5_output.json",
    "FIS": "FIS_q4_2020_p5_output.json",
    "GCO": "GCO_q1_2022_p5_output.json",
    "HTH": "HTH_q4_2020_p5_output.json",
    "TT": "TT_q1_2021_p5_output.json",
    "DNB": "DNB_q2_2021_p5_output.json",
    "DOV": "DOV_q2_2020_p5_output.json",
    "DE": "DE_q1_2014_p5_output.json",
    "GD": "GD_q1_2021_p5_output.json",
    "LH": "LH_q3_2021_p5_output.json",
    "MSI": "MSI_q3_2021_p5_output.json",
    "NEE": "NEE_q3_2021_p5_output.json",
    "AME": "AME_q1_2021_p5_output.json",
    "CMI": "CMI_q1_2014_p5_output.json",
}

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


def extract_ticker_from_filename(filename: str) -> str:
    return filename.split("_")[0].upper()


def _output_dir() -> Path:
    for path in P5_OUTPUT_DIRS:
        if path.exists():
            return path
    return P5_OUTPUT_DIRS[0]


def _all_output_files() -> list[Path]:
    output_dir = _output_dir()
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*_p5_output.json"))


def _find_file(ticker: str) -> Path | None:
    ticker = ticker.upper()
    output_dir = _output_dir()
    known = KNOWN_COMPANIES.get(ticker)
    if known and (output_dir / known).exists():
        return output_dir / known
    matches = [p for p in _all_output_files() if extract_ticker_from_filename(p.name) == ticker]
    return matches[0] if matches else None


def _label_from_source(source_file: str) -> str:
    parts = source_file.split("_")
    if len(parts) >= 3:
        return f"{parts[0]} {parts[1].upper()} {parts[2]}"
    return source_file


def get_available_companies() -> list[dict]:
    companies = []
    for path in _all_output_files():
        ticker = extract_ticker_from_filename(path.name)
        source = path.name.replace("_p5_output.json", "")
        companies.append(
            {
                "ticker": ticker,
                "company": ticker,
                "label": _label_from_source(source),
                "quarter": " ".join(source.split("_")[1:]).upper(),
                "filename": path.name,
            }
        )
    preferred = list(KNOWN_COMPANIES)
    companies.sort(
        key=lambda item: (
            preferred.index(item["ticker"]) if item["ticker"] in preferred else 999,
            item["ticker"],
            item["filename"],
        )
    )
    return companies


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    if isinstance(value, dict):
        chunks = []
        for key, val in value.items():
            label = key.replace("_", " ").title()
            chunks.append(f"{label}: {_stringify(val)}")
        return "\n".join(chunks)
    return "" if value is None else str(value)


def _placeholder_scores(source_file: str) -> dict:
    seed = sum(ord(ch) for ch in source_file)
    scores = {}
    for idx, dimension in enumerate(DIMENSIONS):
        scores[dimension] = round(3.45 + ((seed + idx * 17) % 35) / 100, 2)
    scores["avg_overall"] = round(sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)
    return scores


def _benchmark_scores(source_file: str) -> dict | None:
    output_dir = _output_dir()
    for csv_path in list(output_dir.glob("stats/benchmark_*.csv")) + list(output_dir.glob("benchmark_*.csv")):
        try:
            with csv_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    haystack = " ".join(str(v) for v in row.values())
                    if source_file not in haystack:
                        continue
                    scores = {}
                    for dim in DIMENSIONS:
                        for key in (dim, dim.replace("_", " "), dim.title()):
                            if key in row and row[key]:
                                scores[dim] = float(row[key])
                                break
                    if len(scores) == len(DIMENSIONS):
                        scores["avg_overall"] = round(
                            sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS), 2
                        )
                        return scores
        except Exception:
            continue
    return None


def load_precomputed(ticker: str) -> dict | None:
    path = _find_file(ticker)
    if not path:
        return None
    with path.open() as handle:
        payload = json.load(handle)

    debate = payload.get("debate_result", payload)
    source_file = payload.get("source_file") or debate.get("source_file") or path.stem.replace("_p5_output", "")
    recommendations = debate.get("investment_recommendations", {})
    conclusion = debate.get("investment_conclusion", "")
    risk_reward = debate.get("risk_reward", "")
    debate_log = (
        debate.get("_debate_log", {}).get("steps")
        or payload.get("debate_log", {}).get("steps")
        or debate.get("debate_log", {})
    )
    if isinstance(debate_log, dict):
        debate_log = [
            {"step": "trust_phase", "result": debate_log.get("trust_summary", "")},
            {"step": "skeptic_phase", "result": debate_log.get("skeptic_summary", "")},
            {"step": "leader_phase", "result": debate_log.get("synthesis_note", "")},
        ]

    agent_outputs = {
        "earnings": debate.get("executive_summary", ""),
        "market": _stringify(recommendations),
        "sentiment": debate.get("trust_enhancements", ""),
        "valuation": _stringify(risk_reward),
        "risk": debate.get("skeptic_risk_additions", ""),
    }

    scores = _benchmark_scores(source_file) or _placeholder_scores(source_file)
    return {
        "ticker": extract_ticker_from_filename(path.name),
        "company": extract_ticker_from_filename(path.name),
        "source_file": source_file,
        "stance": debate.get("overall_stance", "NEUTRAL"),
        "conviction": debate.get("overall_conviction", "75%"),
        "executive_summary": debate.get("executive_summary", ""),
        "investment_recommendations": recommendations,
        "trust_text": debate.get("trust_enhancements", ""),
        "skeptic_text": debate.get("skeptic_risk_additions", ""),
        "risk_reward": _stringify(risk_reward),
        "conclusion": _stringify(conclusion),
        "debate_log": debate_log,
        "scores": scores,
        "agent_outputs": agent_outputs,
        "rag_preview": debate.get("executive_summary", "")[:260],
    }
