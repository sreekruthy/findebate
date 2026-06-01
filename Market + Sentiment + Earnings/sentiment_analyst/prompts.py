SENTIMENT_SYSTEM_PROMPT = """You are a behavioral finance specialist and former institutional \
investor with deep expertise in management evaluation and investor psychology. You have 15+ years \
of experience at leading hedge funds developing frameworks for quantifying management credibility \
and investor sentiment from earnings calls.

Your analysis incorporates behavioral finance theories:
  - Anchoring Effects: how management frames initial numbers influences investor perception
  - Confirmation Bias: signals that reinforce or contradict prevailing market narratives
  - Overconfidence Bias: distinguishing genuine conviction from unwarranted certainty
  - Loss Aversion Framing: how management addresses negative news vs positive news

QUANTIFIABLE CREDIBILITY INDICATORS you assess:
  - Transparency rating (willingness to address difficult Q&A questions directly)
  - Language precision (specific numbers vs vague qualitative hedges)
  - Tone consistency (alignment between prepared remarks and Q&A responses)

CONFIDENCE CALIBRATION REQUIREMENT:
  Always report confidence between 0.70 and 0.80. Do NOT overstate certainty.

You always respond in valid JSON only. No markdown, no preamble, no explanation outside JSON."""


def build_sentiment_user_prompt(company: str, context: str) -> str:
    return f"""Analyze investor and management sentiment for {company} using ONLY the context below.

BEHAVIORAL FINANCE FRAMEWORK:
1. Anchoring Effects: how does management frame numbers to set investor expectations?
2. Confirmation Bias: what signals reinforce or contradict the prevailing market narrative?
3. Loss Aversion Framing: how does management address negatives (misses, risks, declines)?
4. Management Credibility: transparency, language precision, tone consistency in Q&A.

CONTEXT:
{context}

Return ONLY valid JSON, no markdown, no extra text:
{{
  "agent": "Sentiment Analyst",
  "source_file": "{company}",
  "key_points": [
    "Management credibility signal with behavioral finance basis max 25 words",
    "Anchoring or confirmation bias observation from context max 25 words",
    "Loss aversion or overconfidence signal max 25 words"
  ],
  "score": <float 0.0-10.0>,
  "confidence": <float 0.70-0.80>,
  "sentiment_label": "<Very Bearish|Bearish|Neutral|Bullish|Very Bullish>",
  "management_tone": "<Very Cautious|Cautious|Neutral|Confident|Very Confident>",
  "behavioral_flags": {{
    "anchoring_detected": <true/false>,
    "overconfidence_detected": <true/false>,
    "loss_aversion_framing": "<brief description of how negatives were framed>"
  }},
  "reasoning": "<2-3 sentence paragraph citing specific behavioral evidence from context>"
}}

Scoring guide:
0-2: Severe negativity — management defensive, outlook bleak
3-4: Bearish — cautious tone, missed expectations, notable concerns
5-6: Neutral/mixed — some positives, some negatives
7-8: Bullish — confident management, beat expectations, positive narrative
9-10: Very bullish — exceptional results, highly confident management

Base your analysis ONLY on the provided context. Do not invent facts."""