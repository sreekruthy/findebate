"""
debate_engine.py — FinDebate MVP Debate Engine
================================================
Orchestrates the full pipeline:
  1. initialize_rag() once
  2. Run all 4 analyst agents
  3. 2-round debate: Trust → Skeptic → Leader (x2)
  4. Return a single structured dict for Streamlit UI

Usage:
    from debate_engine import run_debate
    result = run_debate("Apple")   # or "Tesla"
"""

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

import google.generativeai as genai

# ── Agent imports ────────────────────────────────────────────────────────────
from earnings_agent import earnings_agent
from market_agent import market_analyst, extract_json
from risk_agent import run_risk_analyst
from sentiment_agent import run_sentiment_analyst

# ── RAG imports ───────────────────────────────────────────────────────────────
from rag_module import initialize_rag, retrieve_filtered

load_dotenv()

# ── Gemini setup ──────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini = genai.GenerativeModel("gemini-2.5-flash")

_RAG_INITIALIZED = False   # guard: call initialize_rag only once per process


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAG helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_rag():
    global _RAG_INITIALIZED
    if not _RAG_INITIALIZED:
        initialize_rag()
        _RAG_INITIALIZED = True


def _get_chunks(company: str, data_type: str, k: int = 4) -> list[str]:
    """Retrieve context chunks for risk / sentiment agents."""
    query_map = {
        "earnings_q1": "revenue growth profitability margins earnings guidance financial performance",
        "news_q1":     "market trends stock movement industry outlook risks sentiment",
    }
    query = query_map.get(data_type, "financial analysis")
    return retrieve_filtered(query=query, company=company, data_type=data_type, k=k)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Run the 4 analyst agents
# ═════════════════════════════════════════════════════════════════════════════

def _run_all_agents(company: str) -> dict:
    """
    Run earnings, market, risk, sentiment agents.
    Returns a dict keyed by agent name with their JSON outputs.
    Errors are caught per-agent so one failure doesn't kill the pipeline.
    """
    results = {}

    # 1. Earnings Analyst (uses RAG internally via rag_module)
    try:
        results["earnings"] = earnings_agent(company)
    except Exception as e:
        results["earnings"] = _agent_error("Earnings Analyst", company, str(e))

    # 2. Market Analyst (uses RAG internally via rag module in market_agent)
    try:
        raw = market_analyst(company)
        results["market"] = extract_json(raw)
    except Exception as e:
        results["market"] = _agent_error("Market Analyst", company, str(e))

    # 3. Risk Analyst (needs chunks passed in)
    try:
        chunks = _get_chunks(company, "news_q1", k=4)
        results["risk"] = run_risk_analyst(company, chunks)
    except Exception as e:
        results["risk"] = _agent_error("Risk Analyst", company, str(e))

    # 4. Sentiment Analyst (needs chunks passed in)
    try:
        # blend earnings + news context for richer sentiment signal
        chunks = _get_chunks(company, "earnings_q1", k=2) + _get_chunks(company, "news_q1", k=2)
        results["sentiment"] = run_sentiment_analyst(company, chunks)
    except Exception as e:
        results["sentiment"] = _agent_error("Sentiment Analyst", company, str(e))

    return results


def _agent_error(agent: str, company: str, msg: str) -> dict:
    return {
        "agent": agent,
        "company": company,
        "score": 5.0,
        "key_points": ["Agent failed — see reasoning"],
        "reasoning": f"Error: {msg}",
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Score normalisation & confidence
# ═════════════════════════════════════════════════════════════════════════════

# Weights for the weighted-average score used as a signal baseline.
# Risk score is inverted (high risk score = low risk = good), so we keep it as-is
# because the risk agent already scores: 10 = safe, 0 = dangerous.
_WEIGHTS = {
    "earnings": 0.30,
    "market":   0.25,
    "risk":     0.25,   # already: 10=safe, 0=dangerous
    "sentiment": 0.20,
}

def _compute_weighted_score(agents: dict) -> float:
    total, weight_sum = 0.0, 0.0
    for key, w in _WEIGHTS.items():
        a = agents.get(key, {})
        s = float(a.get("score", 5.0))
        total += s * w
        weight_sum += w
    return round(total / weight_sum, 2) if weight_sum else 5.0


def _score_to_confidence(score: float) -> float:
    """Map 0-10 score to 0.0-1.0 confidence band."""
    return round(min(max(score / 10.0, 0.0), 1.0), 3)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Debate agents (Trust, Skeptic, Leader) via Gemini
# ═════════════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> str:
    try:
        resp = _gemini.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return json.dumps({"error": str(e)})


def _safe_parse(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"raw": text, "parse_error": True}


# ── Trust Agent ───────────────────────────────────────────────────────────────

def _trust_agent(company: str, report: str, agent_analyses: dict, round_num: int) -> dict:
    prompt = f"""You are the Trust Agent in a financial debate panel for {company} (Round {round_num}).

Your role: STRENGTHEN the existing investment report with supporting evidence.
Rules:
- DO NOT change the directional stance (bullish/bearish)
- DO NOT alter Buy/Sell/Hold recommendations
- ADD supporting evidence, reinforce logic, enhance persuasiveness
- Identify which agent analyses most strongly support the current thesis

CURRENT REPORT:
{report}

AGENT ANALYSES:
{json.dumps(agent_analyses, indent=2)}

Respond ONLY with valid JSON — no markdown, no preamble:
{{
  "role": "Trust Agent",
  "round": {round_num},
  "company": "{company}",
  "supporting_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"],
  "strongest_agents": ["agent name(s) whose data best supports the thesis"],
  "enhanced_reasoning": "2-3 sentences strengthening the investment thesis with specific data",
  "confidence_boost": <float 0.0-1.0 indicating how much this evidence strengthens conviction>,
  "enhanced_report_section": "A short enhanced paragraph to add to the report"
}}"""
    return _safe_parse(_call_gemini(prompt))


# ── Skeptic Agent ─────────────────────────────────────────────────────────────

def _skeptic_agent(company: str, trust_output: dict, agent_analyses: dict, round_num: int) -> dict:
    prompt = f"""You are the Skeptic Agent in a financial debate panel for {company} (Round {round_num}).

Your role: IDENTIFY contradictions, risks, and weaknesses in the analysis.
Rules:
- DO NOT flip the final Buy/Sell/Hold — only flag concerns
- DO NOT contradict for the sake of it — be evidence-based
- Look for: agent contradictions, overconfidence, missing risks, data inconsistencies
- Suggest risk mitigations that strengthen (not undermine) the investment case

TRUST AGENT OUTPUT:
{json.dumps(trust_output, indent=2)}

ALL AGENT ANALYSES:
{json.dumps(agent_analyses, indent=2)}

Respond ONLY with valid JSON — no markdown, no preamble:
{{
  "role": "Skeptic Agent",
  "round": {round_num},
  "company": "{company}",
  "contradictions_found": [
    {{"agents": ["agent A", "agent B"], "conflict": "describe the contradiction", "severity": "High|Medium|Low"}}
  ],
  "overlooked_risks": ["risk 1", "risk 2"],
  "overconfidence_flags": ["any agent that seems overconfident and why"],
  "risk_mitigations": ["mitigation 1 that still supports the thesis", "mitigation 2"],
  "skeptic_verdict": "1-2 sentences: overall concern level and what the Leader must address",
  "concern_level": "<one of: Critical | High | Moderate | Low>"
}}"""
    return _safe_parse(_call_gemini(prompt))


# ── Leader Agent ──────────────────────────────────────────────────────────────

def _leader_agent(
    company: str,
    original_report: str,
    trust_output: dict,
    skeptic_output: dict,
    agent_analyses: dict,
    round_num: int,
    is_final: bool,
) -> dict:
    final_flag = "This is the FINAL round — produce the definitive investment report." if is_final else f"This is Round {round_num} — produce an improved intermediate report."

    prompt = f"""You are the Leader Agent synthesizing a financial debate for {company} (Round {round_num}).

{final_flag}

Your role: Synthesize Trust + Skeptic perspectives into the strongest possible report.
Rules:
- PRESERVE all core directional conclusions from the original report
- INTEGRATE Trust's supporting evidence
- ADDRESS Skeptic's concerns without abandoning the thesis
- OUTPUT a clear BUY / SELL / HOLD decision with conviction level
- Base the decision on argument strength, not just numbers

ORIGINAL REPORT:
{original_report}

TRUST AGENT:
{json.dumps(trust_output, indent=2)}

SKEPTIC AGENT:
{json.dumps(skeptic_output, indent=2)}

ALL AGENT ANALYSES (scores for reference):
Earnings score: {agent_analyses.get('earnings', {}).get('score', 'N/A')}
Market score:   {agent_analyses.get('market', {}).get('score', 'N/A')}
Risk score:     {agent_analyses.get('risk', {}).get('score', 'N/A')} (10=safe, 0=dangerous)
Sentiment score:{agent_analyses.get('sentiment', {}).get('score', 'N/A')}

Respond ONLY with valid JSON — no markdown, no preamble:
{{
  "role": "Leader Agent",
  "round": {round_num},
  "company": "{company}",
  "decision": "<BUY | SELL | HOLD>",
  "conviction": "<Strong | Moderate | Weak>",
  "decision_rationale": "2-3 sentences: why this decision, grounded in agent outputs and debate",
  "trust_integration": "1 sentence: what from Trust strengthened the thesis",
  "skeptic_response": "1 sentence: how the main skeptic concern was addressed",
  "key_investment_thesis": ["thesis point 1", "thesis point 2", "thesis point 3"],
  "primary_risk": "The single most important risk to monitor",
  "time_horizons": {{
    "1_day": "<BUY | SELL | HOLD>",
    "1_week": "<BUY | SELL | HOLD>",
    "1_month": "<BUY | SELL | HOLD>"
  }},
  "report_summary": "3-4 sentence institutional-grade investment summary for this round"
}}"""
    return _safe_parse(_call_gemini(prompt))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Debate loop (2 rounds)
# ═════════════════════════════════════════════════════════════════════════════

def _build_initial_report(company: str, agents: dict) -> str:
    """
    Build a plain-text initial report from the 4 agent outputs
    to seed the debate. This is what Trust/Skeptic/Leader debate over.
    """
    lines = [f"=== INITIAL ANALYST REPORT: {company} ===\n"]
    for key in ("earnings", "market", "risk", "sentiment"):
        a = agents.get(key, {})
        lines.append(f"--- {a.get('agent', key.upper())} ---")
        lines.append(f"Score: {a.get('score', 'N/A')}/10")
        for kp in a.get("key_points", []):
            lines.append(f"  • {kp}")
        lines.append(f"Reasoning: {a.get('reasoning', '')}\n")
    return "\n".join(lines)


def _run_debate(company: str, agents: dict, num_rounds: int = 2) -> dict:
    """
    Execute the multi-round debate.
    Returns structured debate history + final leader output.
    """
    initial_report = _build_initial_report(company, agents)
    current_report = initial_report
    rounds = []

    for r in range(1, num_rounds + 1):
        is_final = (r == num_rounds)

        trust   = _trust_agent(company, current_report, agents, r)
        skeptic = _skeptic_agent(company, trust, agents, r)
        leader  = _leader_agent(company, current_report, trust, skeptic, agents, r, is_final)

        rounds.append({
            "round": r,
            "trust":   trust,
            "skeptic": skeptic,
            "leader":  leader,
        })

        # The leader's report summary becomes the next round's base
        current_report = leader.get("report_summary", current_report)

    return {
        "initial_report": initial_report,
        "rounds": rounds,
        "final_leader": rounds[-1]["leader"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Build the UI-ready output dict
# ═════════════════════════════════════════════════════════════════════════════

def _build_chart_data(agents: dict) -> dict:
    """Pre-compute chart-ready data for Streamlit."""
    radar = {
        "labels": ["Earnings", "Market", "Risk (Safety)", "Sentiment"],
        "scores": [
            float(agents.get("earnings",  {}).get("score", 5.0)),
            float(agents.get("market",    {}).get("score", 5.0)),
            float(agents.get("risk",      {}).get("score", 5.0)),
            float(agents.get("sentiment", {}).get("score", 5.0)),
        ],
    }

    bar = {
        "agents": ["Earnings", "Market", "Risk", "Sentiment"],
        "scores": radar["scores"],
        "weights": [_WEIGHTS["earnings"], _WEIGHTS["market"], _WEIGHTS["risk"], _WEIGHTS["sentiment"]],
        "weighted_contributions": [
            round(radar["scores"][i] * list(_WEIGHTS.values())[i], 3)
            for i in range(4)
        ],
    }

    return {"radar": radar, "bar": bar}


def _build_debate_summary(debate: dict) -> list[dict]:
    """
    Flatten the debate rounds into a timeline list for the UI.
    Each entry is one agent turn across all rounds.
    """
    timeline = []
    for rd in debate["rounds"]:
        r = rd["round"]
        t = rd["trust"]
        s = rd["skeptic"]
        l = rd["leader"]

        timeline.append({
            "round": r,
            "agent": "Trust Agent",
            "stance": "Supportive",
            "key_action": f"Strengthened thesis with {len(t.get('supporting_evidence', []))} evidence points",
            "confidence_boost": t.get("confidence_boost", 0.0),
            "summary": t.get("enhanced_reasoning", t.get("raw", ""))[:300],
        })
        timeline.append({
            "round": r,
            "agent": "Skeptic Agent",
            "stance": "Critical",
            "key_action": f"Flagged {len(s.get('contradictions_found', []))} contradictions, concern level: {s.get('concern_level', 'N/A')}",
            "concern_level": s.get("concern_level", "N/A"),
            "summary": s.get("skeptic_verdict", s.get("raw", ""))[:300],
        })
        timeline.append({
            "round": r,
            "agent": "Leader Agent",
            "stance": "Synthesis",
            "key_action": f"Round {r} decision: {l.get('decision', 'N/A')} ({l.get('conviction', 'N/A')} conviction)",
            "decision": l.get("decision", "N/A"),
            "summary": l.get("report_summary", l.get("raw", ""))[:400],
        })
    return timeline


def _extract_risks(agents: dict, final_leader: dict) -> list[dict]:
    """Collect identified risks from risk agent + leader's primary risk."""
    risks = []
    risk_data = agents.get("risk", {})
    for r in risk_data.get("identified_risks", []):
        risks.append({
            "source": "Risk Analyst",
            "category": r.get("category", "Unknown"),
            "description": r.get("description", ""),
            "severity": r.get("severity", "Medium"),
        })
    primary = final_leader.get("primary_risk")
    if primary:
        risks.append({
            "source": "Leader (Debate)",
            "category": "Synthesized",
            "description": primary,
            "severity": "High",
        })
    return risks


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main public entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_debate(company: str, num_rounds: int = 2) -> dict:
    """
    Full pipeline entry point.

    Args:
        company:    "Apple" or "Tesla"
        num_rounds: number of debate rounds (default 2, MVP)

    Returns:
        A single Python dict with everything the Streamlit UI needs.
    """
    company = company.strip().title()
    _ensure_rag()

    # ── Step 1: Run all 4 analyst agents ─────────────────────────────────────
    print(f"[FinDebate] Running analyst agents for {company}...")
    agents = _run_all_agents(company)

    # ── Step 2: Compute baseline scores ──────────────────────────────────────
    weighted_score = _compute_weighted_score(agents)

    # ── Step 3: Run debate rounds ─────────────────────────────────────────────
    print(f"[FinDebate] Starting {num_rounds}-round debate...")
    debate = _run_debate(company, agents, num_rounds=num_rounds)
    final_leader = debate["final_leader"]

    # ── Step 4: Assemble UI-ready output ──────────────────────────────────────
    output = {
        # ── Meta ─────────────────────────────────────────────────────────────
        "meta": {
            "company": company,
            "generated_at": datetime.now().isoformat(),
            "num_rounds": num_rounds,
            "pipeline": "FinDebate MVP — Earnings + Market + Risk + Sentiment → 2-Round Debate",
        },

        # ── Final Decision (top-level for UI hero section) ───────────────────
        "final_decision": {
            "decision":   final_leader.get("decision", "HOLD"),
            "conviction": final_leader.get("conviction", "Moderate"),
            "rationale":  final_leader.get("decision_rationale", ""),
            "time_horizons": final_leader.get("time_horizons", {
                "1_day": "HOLD", "1_week": "HOLD", "1_month": "HOLD"
            }),
            "investment_thesis": final_leader.get("key_investment_thesis", []),
        },

        # ── Risk Score (UI risk gauge) ────────────────────────────────────────
        "risk_summary": {
            "risk_score":      float(agents.get("risk", {}).get("score", 5.0)),
            "risk_level":      agents.get("risk", {}).get("risk_level", "Moderate"),
            "primary_risk":    final_leader.get("primary_risk", ""),
            "identified_risks": _extract_risks(agents, final_leader),
            "mitigation_factors": agents.get("risk", {}).get("mitigation_factors", []),
        },

        # ── Individual Agent Outputs (UI agent cards) ─────────────────────────
        "agent_outputs": {
            "earnings": {
                "agent":       agents["earnings"].get("agent", "Earnings Analyst"),
                "score":       float(agents["earnings"].get("score", 5.0)),
                "confidence":  _score_to_confidence(float(agents["earnings"].get("score", 5.0))),
                "key_points":  agents["earnings"].get("key_points", []),
                "reasoning":   agents["earnings"].get("reasoning", ""),
                "weight":      _WEIGHTS["earnings"],
            },
            "market": {
                "agent":       agents["market"].get("agent", "Market Analyst"),
                "score":       float(agents["market"].get("score", 5.0)),
                "confidence":  _score_to_confidence(float(agents["market"].get("score", 5.0))),
                "key_points":  agents["market"].get("key_points", []),
                "reasoning":   agents["market"].get("reasoning", ""),
                "weight":      _WEIGHTS["market"],
            },
            "risk": {
                "agent":       agents["risk"].get("agent", "Risk Analyst"),
                "score":       float(agents["risk"].get("score", 5.0)),
                "confidence":  _score_to_confidence(float(agents["risk"].get("score", 5.0))),
                "key_points":  agents["risk"].get("key_points", []),
                "reasoning":   agents["risk"].get("reasoning", ""),
                "risk_level":  agents["risk"].get("risk_level", "Moderate"),
                "weight":      _WEIGHTS["risk"],
            },
            "sentiment": {
                "agent":           agents["sentiment"].get("agent", "Sentiment Analyst"),
                "score":           float(agents["sentiment"].get("score", 5.0)),
                "confidence":      _score_to_confidence(float(agents["sentiment"].get("score", 5.0))),
                "key_points":      agents["sentiment"].get("key_points", []),
                "reasoning":       agents["sentiment"].get("reasoning", ""),
                "sentiment_label": agents["sentiment"].get("sentiment_label", "Neutral"),
                "management_tone": agents["sentiment"].get("management_tone", "Neutral"),
                "weight":          _WEIGHTS["sentiment"],
            },
        },

        # ── Scores for Charts ─────────────────────────────────────────────────
        "scores": {
            "weighted_average": weighted_score,
            "individual": {
                "earnings":  float(agents["earnings"].get("score", 5.0)),
                "market":    float(agents["market"].get("score",   5.0)),
                "risk":      float(agents["risk"].get("score",     5.0)),
                "sentiment": float(agents["sentiment"].get("score",5.0)),
            },
        },

        # ── Chart-ready data ──────────────────────────────────────────────────
        "chart_data": _build_chart_data(agents),

        # ── Debate Summary (UI debate timeline) ───────────────────────────────
        "debate_summary": {
            "timeline":        _build_debate_summary(debate),
            "initial_report":  debate["initial_report"],
            "final_report":    final_leader.get("report_summary", ""),
            "rounds_detail":   [
                {
                    "round": rd["round"],
                    "trust_confidence_boost":  rd["trust"].get("confidence_boost", 0.0),
                    "skeptic_concern_level":   rd["skeptic"].get("concern_level", "N/A"),
                    "leader_decision":         rd["leader"].get("decision", "N/A"),
                    "leader_conviction":       rd["leader"].get("conviction", "N/A"),
                    "contradictions_found":    rd["skeptic"].get("contradictions_found", []),
                    "trust_evidence":          rd["trust"].get("supporting_evidence", []),
                    "skeptic_risks":           rd["skeptic"].get("overlooked_risks", []),
                    "leader_thesis":           rd["leader"].get("key_investment_thesis", []),
                }
                for rd in debate["rounds"]
            ],
        },

        # ── Raw debate rounds (for debugging / advanced UI) ───────────────────
        "_raw_debate_rounds": debate["rounds"],
    }

    print(f"[FinDebate] ✅ Done. Decision: {output['final_decision']['decision']} "
          f"({output['final_decision']['conviction']} conviction)")
    return output


# ═════════════════════════════════════════════════════════════════════════════
# CLI test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    company = sys.argv[1] if len(sys.argv) > 1 else "Apple"
    result = run_debate(company)

    # Save full output for inspection
    out_path = f"debate_output_{company.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[FinDebate] Full output saved → {out_path}")

    # Print decision summary
    fd = result["final_decision"]
    print(f"\n{'='*50}")
    print(f"  COMPANY:    {result['meta']['company']}")
    print(f"  DECISION:   {fd['decision']}")
    print(f"  CONVICTION: {fd['conviction']}")
    print(f"  RATIONALE:  {fd['rationale']}")
    print(f"  RISK LEVEL: {result['risk_summary']['risk_level']}")
    print(f"  W. SCORE:   {result['scores']['weighted_average']}/10")
    print(f"{'='*50}")