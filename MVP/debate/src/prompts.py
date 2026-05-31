"""
FinDebate — Person 5
Agent system prompts (verbatim from paper Appendix D, Figures 11-13)
and user-prompt builders that format the P4 synthesis as readable prose.
"""

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS  (Appendix D, paper)
# ══════════════════════════════════════════════════════════════════════════════

TRUST_SYSTEM_PROMPT = """You are the Trust agent in a professional investment evaluation. Your task is to PRESERVE and ENHANCE the existing investment analysis while maintaining its core structure and recommendations.

CRITICAL REQUIREMENTS FOR PROFESSIONAL STANDARDS:
- PRESERVE all existing Long/Short recommendations for 1-day, 1-week, and 1-month timeframes
- MAINTAIN the persuasive tone and conviction levels already established
- ENHANCE the supporting evidence and rationale WITHOUT changing core conclusions
- KEEP all specific catalysts, timelines, and actionable insights already provided
- DO NOT remove or weaken any professional investment guidance elements

Your responsibilities:
- Strengthen existing arguments with additional supporting evidence
- Enhance the persuasive power of existing recommendations
- Add complementary insights that support the existing investment thesis
- Maintain professional investment language and structure
- NEVER contradict or weaken the existing Long/Short recommendations

Response format: Provide enhanced analysis that makes the existing investment recommendations MORE persuasive while preserving all core elements."""


SKEPTIC_SYSTEM_PROMPT = """You are the Skeptic agent in a professional investment evaluation. Your task is to identify potential risks and strengthen the analysis through critical examination, while PRESERVING the core investment recommendations.

CRITICAL REQUIREMENTS FOR PROFESSIONAL STANDARDS:
- DO NOT change or contradict existing Long/Short recommendations for any timeframe
- IDENTIFY risks and challenges to STRENGTHEN risk management sections
- ENHANCE risk-reward balance discussions without undermining confidence
- ADD risk mitigation strategies that support the investment thesis
- MAINTAIN the persuasive power for investor decision-making

Your responsibilities:
- Identify potential risks that should be acknowledged in risk management
- Suggest risk mitigation strategies that strengthen the investment case
- Enhance scenario analysis with balanced risk-reward assessment
- Strengthen the analysis by addressing potential investor concerns
- PRESERVE all existing timeframe recommendations and conviction levels

Response format: Provide critical analysis that STRENGTHENS the investment recommendations by addressing risks and enhancing credibility."""


LEADER_SYSTEM_PROMPT = """You are the Leader agent in a professional investment evaluation. Your task is to create the FINAL OPTIMIZED REPORT that maximizes investor persuasion while preserving all critical professional elements.

CRITICAL REQUIREMENTS FOR PROFESSIONAL STANDARDS:
This report will be used by professional investors who will make Long/Short investment decisions based on YOUR analysis for 1-day, 1-week, and 1-month periods. Your success depends on providing accurate, actionable guidance.

MANDATORY ELEMENTS TO PRESERVE:
- ALL existing Long/Short recommendations for each timeframe with conviction levels
- ALL persuasive evidence and investment rationale
- ALL specific catalysts, timelines, and actionable insights
- ALL professional investment guidance and implementation steps
- CLEAR multi-timeframe investment strategy sections

Your responsibilities:
- Synthesize Trust and Skeptic perspectives into ONE FINAL OPTIMIZED REPORT
- MAXIMIZE persuasive power for investor decision-making
- PRESERVE all existing investment recommendations and enhance their supporting evidence
- MAINTAIN professional investment report structure and flow
- ENSURE professional investors will be convinced to follow the investment guidance

Response format: Provide the FINAL OPTIMIZED INVESTMENT REPORT that preserves all critical elements while maximizing persuasive impact for professional investment decisions.

IMPORTANT — Output your response as valid JSON matching this exact schema:
{
  "overall_stance": "<BULLISH|BEARISH|NEUTRAL>",
  "overall_conviction": "<percentage, e.g. 75%>",
  "executive_summary": "<string>",
  "investment_recommendations": {
    "one_day":   {"position": "<LONG|SHORT|NEUTRAL>", "conviction": "<pct>", "rationale": "<string>"},
    "one_week":  {"position": "<LONG|SHORT|NEUTRAL>", "conviction": "<pct>", "rationale": "<string>"},
    "one_month": {"position": "<LONG|SHORT|NEUTRAL>", "conviction": "<pct>", "rationale": "<string>"}
  },
  "trust_enhancements": "<key evidence additions from Trust Agent>",
  "skeptic_risk_additions": "<key risk additions from Skeptic Agent>",
  "risk_reward": {
    "upside_catalysts": ["<string>", "..."],
    "downside_risks":   ["<string>", "..."],
    "position_sizing":  "<string>",
    "hedge_strategies": ["<string>", "..."]
  },
  "investment_conclusion": {
    "final_stance": "<BULLISH|BEARISH|NEUTRAL>",
    "conviction": "<pct>",
    "top_3_insights": ["<string>", "<string>", "<string>"]
  },
  "debate_log": {
    "trust_summary":   "<one-sentence summary of Trust Agent additions>",
    "skeptic_summary": "<one-sentence summary of Skeptic Agent additions>",
    "synthesis_note":  "<how Leader reconciled both perspectives>"
  }
}
Output ONLY the JSON object. No markdown fences, no preamble."""


# ══════════════════════════════════════════════════════════════════════════════
# USER-PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_synthesis_as_prose(synthesis: dict) -> str:
    """
    Convert the P4 synthesis JSON into a readable investment report string
    that Trust / Skeptic agents can work with naturally.
    """
    rec = synthesis.get("investment_recommendations", {})
    one_day   = rec.get("one_day",   {})
    one_week  = rec.get("one_week",  {})
    one_month = rec.get("one_month", {})

    conc = synthesis.get("investment_conclusion", {})
    insights = conc.get("top_3_insights", [])
    insight_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(insights))

    rr = synthesis.get("risk_reward", {})
    up = "\n".join(f"  - {c}" for c in rr.get("upside_catalysts", []))
    dn = "\n".join(f"  - {r}" for r in rr.get("downside_risks", []))
    hg = "\n".join(f"  - {h}" for h in rr.get("hedge_strategies", []))

    mds = synthesis.get("multi_dimensional_synthesis", {})

    report = f"""
════════════════════════════════════════════════════════════
INVESTMENT ANALYSIS REPORT — {synthesis.get('source_file', 'UNKNOWN')}
════════════════════════════════════════════════════════════

OVERALL STANCE:    {synthesis.get('overall_stance', 'N/A')}
CONVICTION LEVEL:  {synthesis.get('overall_conviction', 'N/A')}

─── EXECUTIVE SUMMARY ──────────────────────────────────────
{synthesis.get('executive_summary', '')}

─── MULTI-DIMENSIONAL ANALYSIS ─────────────────────────────
Earnings Highlights:
  {mds.get('earnings_highlights', '')}

Market Positioning:
  {mds.get('market_positioning', '')}

Management Sentiment:
  {mds.get('management_sentiment', '')}

Valuation Summary:
  {mds.get('valuation_summary', '')}

Risk Profile:
  {mds.get('risk_profile', '')}

─── INVESTMENT RECOMMENDATIONS ─────────────────────────────
1-DAY  | {one_day.get('position','N/A')} | Conviction: {one_day.get('conviction','N/A')}
  {one_day.get('expected_direction', one_day.get('rationale', ''))}

1-WEEK | {one_week.get('position','N/A')} | Conviction: {one_week.get('conviction','N/A')}
  {one_week.get('expected_direction', one_week.get('rationale', ''))}

1-MONTH| {one_month.get('position','N/A')} | Conviction: {one_month.get('conviction','N/A')}
  {one_month.get('expected_direction', one_month.get('rationale', ''))}

─── RISK / REWARD ──────────────────────────────────────────
Upside Catalysts:
{up}

Downside Risks:
{dn}

Position Sizing:  {rr.get('position_sizing', 'N/A')}
Hedge Strategies:
{hg}

─── TOP INSIGHTS ───────────────────────────────────────────
{insight_text}

─── REASONING ──────────────────────────────────────────────
{synthesis.get('reasoning', '')}
════════════════════════════════════════════════════════════
""".strip()
    return report


def _format_agent_analysis(p3_data: dict | None, p4_data: dict) -> str:
    """
    Summarise supporting agent outputs (P3 + P4 non-synthesis agents)
    into a short context block for the debate agents.
    """
    lines = ["─── SUPPORTING AGENT ANALYSIS (CONTEXT) ───────────────────"]

    if p3_data:
        agents3 = p3_data.get("agents", {})
        for name, ag in agents3.items():
            pts = ag.get("key_points", [])
            if pts:
                lines.append(f"\n[{name.upper()} ANALYST]")
                for p in pts[:4]:           # top 4 key points
                    lines.append(f"  • {p}")
                lines.append(f"  Score: {ag.get('score', 'N/A')}")

    agents4 = p4_data.get("agents", {})
    for name in ("valuation", "risk"):
        ag = agents4.get(name, {})
        # valuation is stored as raw JSON string; risk is a dict
        if isinstance(ag, dict) and "key_points" in ag:
            lines.append(f"\n[{name.upper()} ANALYST]")
            for p in ag.get("key_points", [])[:4]:
                lines.append(f"  • {p}")
            lines.append(f"  Risk Rating: {ag.get('overall_risk_rating', ag.get('investment_stance', 'N/A'))}")
        elif isinstance(ag, dict) and "raw" in ag:
            import json as _json
            try:
                raw = _json.loads(ag["raw"])
                lines.append(f"\n[{name.upper()} ANALYST]")
                for p in raw.get("key_points", [])[:4]:
                    lines.append(f"  • {p}")
                lines.append(f"  Stance: {raw.get('investment_stance', 'N/A')}")
            except Exception:
                pass

    return "\n".join(lines)


def build_trust_prompt(synthesis: dict, p3_data: dict | None, p4_data: dict) -> str:
    prose   = _format_synthesis_as_prose(synthesis)
    context = _format_agent_analysis(p3_data, p4_data)
    return f"""Below is the original investment analysis report that you must ENHANCE while strictly preserving all Long/Short recommendations and conviction levels.

{prose}

{context}

Your task:
1. Strengthen the existing evidence and arguments — do NOT change any direction or recommendation.
2. Add supporting data points, financial rationale, or market context that reinforces the current stance.
3. Improve linguistic clarity and persuasiveness.
4. Return your enhanced version of the report in clear prose (not JSON — the Leader Agent will handle final structuring)."""


def build_skeptic_prompt(synthesis: dict, trust_enhanced: str,
                          p3_data: dict | None, p4_data: dict) -> str:
    prose   = _format_synthesis_as_prose(synthesis)
    context = _format_agent_analysis(p3_data, p4_data)
    return f"""Below is the original investment analysis report and the Trust Agent's enhanced version. Your job is to critically examine both and ADD risk factors and hedging strategies — while NEVER changing the Long/Short recommendations.

─── ORIGINAL REPORT ────────────────────────────────────────
{prose}

─── TRUST AGENT ENHANCED VERSION ───────────────────────────
{trust_enhanced}

{context}

Your task:
1. Identify material risk factors NOT yet addressed (or under-addressed) in the report.
2. Suggest concrete hedge strategies that protect the position without contradicting the investment direction.
3. Improve the scenario analysis (bull/base/bear) for balance.
4. PRESERVE all Long/Short recommendations and conviction levels — you may only ADD risk language, not remove or flip anything.
5. Return your risk-augmented version in clear prose (not JSON — the Leader Agent will handle final structuring)."""


def build_leader_prompt(synthesis: dict, trust_enhanced: str,
                         skeptic_enhanced: str) -> str:
    prose = _format_synthesis_as_prose(synthesis)
    # Extract original recommendations for safety check reference
    rec = synthesis.get("investment_recommendations", {})
    orig_1d  = rec.get("one_day",   {}).get("position", "N/A")
    orig_1w  = rec.get("one_week",  {}).get("position", "N/A")
    orig_1m  = rec.get("one_month", {}).get("position", "N/A")
    orig_stance = synthesis.get("overall_stance", "N/A")

    return f"""You are the Leader Agent. Synthesize the Trust Agent and Skeptic Agent outputs into ONE final optimized investment report.

─── ORIGINAL REPORT (source of truth for recommendations) ──
{prose}

─── TRUST AGENT OUTPUT ─────────────────────────────────────
{trust_enhanced}

─── SKEPTIC AGENT OUTPUT ───────────────────────────────────
{skeptic_enhanced}

MANDATORY CONSTRAINTS — these must be preserved verbatim in your output:
  • Overall Stance  : {orig_stance}
  • 1-Day  Position : {orig_1d}
  • 1-Week Position : {orig_1w}
  • 1-Month Position: {orig_1m}

Synthesize the best of both perspectives into the most compelling, professionally written institutional-grade investment report possible. Output ONLY the JSON object described in your system prompt."""
