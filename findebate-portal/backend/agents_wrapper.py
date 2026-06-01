import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from rag_wrapper import chunks_to_text, get_agent_context, merge_chunks


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

P3_PATHS = [
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/Market + Sentiment + Earnings",
    "/Users/sreekruthyreddy/Documents/GitHub/findebate",
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/MVP",
    os.path.expanduser("~/findebate/findebate-main/findebate-main"),
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/Downloads/findebate-main/findebate-main",
]
P4_PATHS = [
    os.path.expanduser("~/findebate/P4_Analyst/P4_Analyst"),
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/P4_Analyst/P4_Analyst",
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/P4_Analyst",
    "/Users/sreekruthyreddy/Documents/GitHub/findebate/Valuation + Risk + Report Synthesis",
]

PIPELINE_AVAILABLE = False
PIPELINE_ERROR = ""
_gemini_client = None
_gemini_timestamps = []


def _disable_import_time_nltk_downloads() -> None:
    if os.getenv("FINDEBATE_ALLOW_NLTK_DOWNLOADS", "").strip() == "1":
        return
    try:
        import nltk

        nltk.download = lambda *args, **kwargs: True
    except Exception:
        pass


def _add_paths(paths: list[str]) -> None:
    for base in paths:
        if os.path.isdir(base) and base not in sys.path:
            sys.path.insert(0, base)
        for child in ("Earnings_Analyst", "Market_Analyst"):
            path = os.path.join(base, child)
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)


_add_paths(P3_PATHS)
_disable_import_time_nltk_downloads()


def _install_p3_import_shims() -> None:
    import types

    if "MVP" not in sys.modules:
        mvp_module = types.ModuleType("MVP")
        mvp_module.__path__ = []
        sys.modules["MVP"] = mvp_module
    if "MVP.rag_module" not in sys.modules:
        rag_module = types.ModuleType("MVP.rag_module")
        rag_module.initialize_rag = lambda *args, **kwargs: None
        rag_module.retrieve_filtered = lambda *args, **kwargs: []
        sys.modules["MVP.rag_module"] = rag_module


_install_p3_import_shims()

try:
    from groq import Groq
    from earnings_agent import EARNINGS_SYSTEM_PROMPT, build_earnings_user_prompt
    from market_agent import MARKET_SYSTEM_PROMPT, build_market_user_prompt

    try:
        from sentiment_analyst.prompts import SENTIMENT_SYSTEM_PROMPT, build_sentiment_user_prompt
    except Exception:
        from MVP.sentiment_analyst.prompts import SENTIMENT_SYSTEM_PROMPT, build_sentiment_user_prompt

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    PIPELINE_AVAILABLE = bool(os.getenv("GROQ_API_KEY") and os.getenv("GEMINI_API_KEY"))
except Exception as exc:
    PIPELINE_ERROR = str(exc)


def _safe_parse(text: str) -> dict:
    import json

    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(clean[start:end])
        return {"error": "parse_failed", "raw": clean[:500]}


def _call_groq(system_prompt: str, user_prompt: str) -> dict:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    return _safe_parse(response.choices[0].message.content.strip())


def _call_groq_json(system_prompt: str, user_prompt: str, max_tokens: int = 5000) -> dict:
    if not os.getenv("GROQ_API_KEY"):
        return {"error": "groq_key_missing", "raw": "GROQ_API_KEY is missing"}
    response = groq_client.chat.completions.create(
        model=os.getenv("FINDEBATE_GROQ_REPORT_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        top_p=0.85,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return _safe_parse(response.choices[0].message.content.strip())


def _wait_for_gemini_rate_limit() -> None:
    global _gemini_timestamps
    now = time.time()
    _gemini_timestamps = [stamp for stamp in _gemini_timestamps if now - stamp < 60]
    if len(_gemini_timestamps) >= 8:
        time.sleep(60 - (now - _gemini_timestamps[0]) + 2)
        now = time.time()
        _gemini_timestamps = [stamp for stamp in _gemini_timestamps if now - stamp < 60]
    _gemini_timestamps.append(time.time())


def _call_gemini(system_prompt: str, user_prompt: str, max_tokens: int = 5000) -> dict:
    global _gemini_client
    if os.getenv("FINDEBATE_P4_PROVIDER", "").strip().lower() == "groq":
        return _call_groq_json(system_prompt, user_prompt, max_tokens=max_tokens)
    if _gemini_client is None:
        from google import genai
        from google.genai import types

        _gemini_client = (
            genai.Client(api_key=os.getenv("GEMINI_API_KEY")),
            types,
        )
    client, types = _gemini_client
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    last_error = None
    for attempt in range(3):
        _wait_for_gemini_rate_limit()
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    top_p=0.85,
                    max_output_tokens=max_tokens,
                    http_options=types.HttpOptions(timeout=120000),
                ),
            )
            return _safe_parse(response.text or "")
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "429" in message or "quota" in message or "rate" in message:
                if os.getenv("GROQ_API_KEY"):
                    return _call_groq_json(system_prompt, user_prompt, max_tokens=max_tokens)
                time.sleep(30 * (attempt + 1))
            else:
                time.sleep(10)
    if os.getenv("GROQ_API_KEY"):
        return _call_groq_json(system_prompt, user_prompt, max_tokens=max_tokens)
    return {"error": "gemini_failed", "raw": str(last_error)}


def run_earnings(source_file: str, context_str: str) -> dict:
    result = _call_groq(EARNINGS_SYSTEM_PROMPT, build_earnings_user_prompt(source_file, context_str))
    result.setdefault("agent", "Earnings Analyst")
    result.setdefault("source_file", source_file)
    return result


def run_market(source_file: str, context_str: str) -> dict:
    result = _call_groq(MARKET_SYSTEM_PROMPT, build_market_user_prompt(source_file, context_str))
    result.setdefault("agent", "Market Analyst")
    result.setdefault("source_file", source_file)
    return result


def run_sentiment(source_file: str, context_str: str) -> dict:
    result = _call_groq(SENTIMENT_SYSTEM_PROMPT, build_sentiment_user_prompt(source_file, context_str))
    result.setdefault("agent", "Sentiment Analyst")
    result.setdefault("source_file", source_file)
    return result


def run_sentiment_from_chunks(source_file: str, chunks: list) -> dict:
    return run_sentiment(source_file, chunks_to_text(chunks))


def run_valuation_agent(source_file: str, top_k: int = 5, fallback_chunks: list | None = None) -> dict:
    chunks = get_agent_context(source_file=source_file, top_k=top_k).get("valuation_agent", [])
    chunks = merge_chunks(fallback_chunks or [], chunks, limit=top_k)
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
    "source_file": "{source_file}",
    "timestamp": "{datetime.now().isoformat()}",
    "investment_stance": "UNDERVALUED or FAIRLY VALUED or OVERVALUED",
    "conviction_level": "75%",
    "key_points": ["Key valuation insight 1", "Key valuation insight 2", "Key valuation insight 3", "Key valuation insight 4", "Key valuation insight 5"],
    "dcf_signals": {{"growth_outlook": "brief growth assessment", "margin_trajectory": "brief margin assessment", "capital_efficiency": "brief capital allocation assessment"}},
    "scenarios": {{"bull_case": "brief bull case", "base_case": "brief base case", "bear_case": "brief bear case"}},
    "score": 7.5,
    "reasoning": "Concise paragraph justifying stance and score"
}}"""
    result = _call_gemini(system_prompt, user_prompt, max_tokens=5000)
    result.setdefault("agent", "Valuation Analyst")
    result.setdefault("source_file", source_file)
    return result


def run_risk_agent(source_file: str, top_k: int = 5, fallback_chunks: list | None = None) -> dict:
    chunks = get_agent_context(source_file=source_file, top_k=top_k).get("risk_agent", [])
    chunks = merge_chunks(fallback_chunks or [], chunks, limit=top_k)
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
    "source_file": "{source_file}",
    "timestamp": "{datetime.now().isoformat()}",
    "overall_risk_rating": "LOW or MODERATE or HIGH or VERY HIGH",
    "conviction_level": "75%",
    "key_points": ["Key risk insight 1", "Key risk insight 2", "Key risk insight 3", "Key risk insight 4", "Key risk insight 5"],
    "risk_dimensions": {{"credit_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}}, "interest_rate_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}}, "liquidity_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}}, "operational_risk": {{"rating": "LOW or MODERATE or HIGH", "assessment": "brief"}}}},
    "position_sizing": {{"recommended_position": "X% of portfolio", "max_position": "X% of portfolio", "hedge_strategies": ["strategy 1", "strategy 2"]}},
    "risk_triggers": ["Trigger 1", "Trigger 2", "Trigger 3"],
    "score": 7.5,
    "reasoning": "Concise paragraph justifying overall risk rating"
}}"""
    result = _call_gemini(system_prompt, user_prompt, max_tokens=5000)
    result.setdefault("agent", "Risk Analyst")
    result.setdefault("source_file", source_file)
    return result


def run_report_synthesizer(agent_outputs: list[dict], source_file: str) -> dict:
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

Synthesize these analyst outputs into a single institutional investment report.

ANALYST OUTPUTS:
{agents_summary}

FIELD MAPPING INSTRUCTIONS:
- earnings_highlights -> use Earnings Analyst output
- market_positioning -> use Market Predictor output
- management_sentiment -> use Sentiment Analyst output
- valuation_summary -> use Valuation Analyst output
- risk_profile -> use Risk Analyst output

IMPORTANT: Do NOT default to NEUTRAL. Make a decisive BULLISH or BEARISH call based on the evidence.

Return ONLY this exact JSON structure:
{{
    "agent": "Report Synthesizer",
    "source_file": "{source_file}",
    "timestamp": "{datetime.now().isoformat()}",
    "overall_stance": "BULLISH or NEUTRAL or BEARISH",
    "overall_conviction": "75%",
    "executive_summary": "150 word summary synthesizing all available agent perspectives",
    "multi_dimensional_synthesis": {{"earnings_highlights": "text", "market_positioning": "text", "management_sentiment": "text", "valuation_summary": "text", "risk_profile": "text"}},
    "investment_recommendations": {{
        "one_day": {{"position": "LONG or SHORT or NEUTRAL", "conviction": "75%", "rationale": "brief rationale"}},
        "one_week": {{"position": "LONG or SHORT or NEUTRAL", "conviction": "75%", "rationale": "brief rationale"}},
        "one_month": {{"position": "LONG or SHORT or NEUTRAL", "conviction": "75%", "rationale": "brief rationale"}}
    }},
    "risk_reward": {{"upside_catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"], "downside_risks": ["risk 1", "risk 2", "risk 3"], "position_sizing": "X% of portfolio", "hedge_strategies": ["hedge 1", "hedge 2"]}},
    "investment_conclusion": {{"final_stance": "BULLISH or NEUTRAL or BEARISH", "conviction": "75%", "top_3_insights": ["insight 1", "insight 2", "insight 3"]}},
    "agent_scores_summary": {{"valuation_score": 7.5, "risk_score": 7.5, "composite_score": 7.5}},
    "reasoning": "Concise paragraph explaining how all agent perspectives were synthesized"
}}"""
    result = _call_gemini(system_prompt, user_prompt, max_tokens=8000)
    result.setdefault("agent", "Report Synthesizer")
    result.setdefault("source_file", source_file)
    return result


async def run_p3_agents(source_file: str, context: dict) -> dict:
    loop = asyncio.get_event_loop()
    earnings_ctx = chunks_to_text(context.get("earnings_agent", []))
    market_ctx = chunks_to_text(context.get("market_agent", []))
    sentiment_ctx = chunks_to_text(context.get("sentiment_agent", []))
    earnings, market, sentiment = await asyncio.gather(
        loop.run_in_executor(None, run_earnings, source_file, earnings_ctx),
        loop.run_in_executor(None, run_market, source_file, market_ctx),
        loop.run_in_executor(None, run_sentiment, source_file, sentiment_ctx),
    )
    return {"earnings": earnings, "market": market, "sentiment": sentiment}


async def run_p4_agents(source_file: str, p3_outputs: dict | None = None, fallback_chunks: list | None = None) -> dict:
    loop = asyncio.get_event_loop()
    valuation, risk = await asyncio.gather(
        loop.run_in_executor(None, run_valuation_agent, source_file, 5, fallback_chunks),
        loop.run_in_executor(None, run_risk_agent, source_file, 5, fallback_chunks),
    )
    p3_outputs = p3_outputs or {}
    all_outputs = [
        p3_outputs.get("earnings"),
        p3_outputs.get("market"),
        p3_outputs.get("sentiment"),
        valuation,
        risk,
    ]
    all_outputs = [output for output in all_outputs if output]
    synthesis = await loop.run_in_executor(None, run_report_synthesizer, all_outputs, source_file)
    return {"valuation": valuation, "risk": risk, "synthesis": synthesis}
