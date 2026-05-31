RISK_SYSTEM_PROMPT = """You are a senior risk management specialist and former institutional portfolio manager \
with extensive experience in equity risk assessment for major asset management firms. \
Your role is to identify, categorize, and score material risks from financial documents \
including earnings calls, news articles, and analyst reports.

Your analysis must be:
- Balanced — avoid both excessive pessimism and unwarranted optimism
- Grounded strictly in the provided context (no hallucination)
- Focused on risks that are material to investment outcomes
- Supported by specific evidence from the context

You always respond in valid JSON only. No markdown, no preamble, no explanation outside JSON."""


def build_risk_user_prompt(company: str, context: str) -> str:
    return f"""Perform a comprehensive risk assessment for {company} based on the context below.

CONTEXT:
{context}

You must respond with ONLY a JSON object in this exact format:
{{
  "agent": "Risk Analyst",
  "company": "{company}",
  "key_points": [
    "Specific risk point 1 with evidence from context",
    "Specific risk point 2 ...",
    "Specific risk point 3 ..."
  ],
  "score": <float between 0.0 and 10.0, where 0=extreme risk/danger, 10=very low risk/safe>,
  "risk_level": "<one of: Critical | High | Moderate | Low | Minimal>",
  "identified_risks": [
    {{
      "category": "<risk category>",
      "description": "<1-2 sentence description of this specific risk>",
      "severity": "<one of: High | Medium | Low>"
    }}
  ],
  "mitigation_factors": [
    "Factor that reduces or offsets a risk (from context)",
    "..."
  ],
  "reasoning": "<2-3 sentence paragraph summarizing overall risk posture with evidence>"
}}

Risk score guide:
0-2: Critical risk — multiple severe risks, existential threats, no mitigation
3-4: High risk — significant risks outweigh mitigations, near-term impact likely
5-6: Moderate risk — meaningful risks but balanced by company strengths
7-8: Low risk — limited risks, strong mitigation, company well-positioned
9-10: Minimal risk — very stable, risks are minor or well-managed

Include 2-5 identified risks. For mitigation_factors, list 1-3 concrete things from the
context that reduce the risk.

Base your analysis ONLY on the provided context. Do not invent facts."""