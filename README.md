# FinDebate — Multi-Agent Financial Analysis System

An MVP implementation of the [FinDebate research paper](https://arxiv.org/abs/2509.17395) — a multi-agent debate framework for institutional-grade financial analysis. Built for Apple and Tesla using Q1 2025 earnings and news data.

---

## What It Does

Four specialized AI analysts (Earnings, Market, Risk, Sentiment) independently analyze a company, then a three-agent debate loop (Trust → Skeptic → Leader) refines their conclusions into a final **BUY / SELL / HOLD** investment decision with full reasoning and risk assessment.

---

## Project Structure

```
FINDEBATE/
├── debate_engine.py            ← Main orchestrator — run this
├── rag_module.py               ← Shared RAG (ChromaDB + embeddings)
├── streamlit_ui.py             ← Streamlit dashboard (UI)
├── .env                        ← API keys (not committed)
├── data/
│   ├── apple/                  ← Apple Q1 2025 cleaned text files
│   └── tesla/                  ← Tesla Q1 2025 cleaned text files
├── Earnings_Analyst/
│   ├── earnings_agent.py
│   └── rag_module.py           ← Local RAG for earnings
├── Market_Analyst/
│   ├── market_agent.py
│   ├── rag_module.py           ← Local RAG for market
│   └── rag/
├── risk_analyst/
│   ├── risk_agent.py
│   ├── prompts.py
│   └── __init__.py
├── sentiment_analyst/
│   ├── sentiment_agent.py
│   ├── prompts.py
│   └── __init__.py
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/findebate.git
cd findebate
```

### 2. Install dependencies

```bash
pip install google-genai groq python-dotenv sentence-transformers chromadb nltk streamlit plotly
```

### 3. Create your `.env` file

Create a file called `.env` in the root `FINDEBATE/` folder:

```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

**Getting API keys:**
- **Gemini** → https://aistudio.google.com/app/apikey (free tier available)
- **Groq** → https://console.groq.com/keys (free, very fast)

---

## Running the Project

### Option A — Test the debate engine directly (CLI)

```bash
python3 debate_engine.py Apple
python3 debate_engine.py Tesla
```

This runs the full pipeline and saves a `debate_output_apple.json` for inspection. Expected runtime: **2–5 minutes** (first run takes longer due to ChromaDB initialization).

### Option B — Run the Streamlit UI

```bash
streamlit run streamlit_ui.py
```

Then open `http://localhost:8501` in your browser. Select Apple or Tesla, click Run Analysis, and wait for results.

---

## How It Works

### Pipeline

```
User selects company (Apple / Tesla)
            ↓
initialize_rag()          — embed data into ChromaDB (once per session)
            ↓
┌─────────────────────────────────────┐
│         4 Analyst Agents            │
│  Earnings  Market  Risk  Sentiment  │
│  (Gemini) (Gemini)(Groq) (Groq)    │
└──────────────┬──────────────────────┘
               ↓
        Initial Report
               ↓
    ┌──── Round 1 Debate ────┐
    │  Trust Agent           │  ← strengthens thesis
    │  Skeptic Agent         │  ← flags contradictions & risks
    │  Leader Agent          │  ← synthesizes → interim decision
    └────────────────────────┘
               ↓
    ┌──── Round 2 Debate ────┐
    │  Trust Agent           │
    │  Skeptic Agent         │
    │  Leader Agent          │  ← final BUY / SELL / HOLD
    └────────────────────────┘
               ↓
        Python dict returned to Streamlit UI
```

### Agents

| Agent | Model | Role |
|---|---|---|
| Earnings Analyst | Gemini 2.5 Flash | Revenue, profitability, margins, guidance |
| Market Analyst | Gemini 2.5 Flash | Market trends, stock movement, industry outlook |
| Risk Analyst | Groq / Llama 3.1 8B | Risk identification, severity scoring |
| Sentiment Analyst | Groq / Llama 3.1 8B | Management tone, investor sentiment |
| Trust Agent | Gemini 2.5 Flash | Strengthens the investment thesis |
| Skeptic Agent | Gemini 2.5 Flash | Challenges contradictions, flags overlooked risks |
| Leader Agent | Gemini 2.5 Flash | Final synthesis → BUY / SELL / HOLD |


## Output Structure (Python Dict)

`run_debate("Apple")` returns a single dict with everything the UI needs:

```
result
├── meta                     pipeline info, timestamp
├── final_decision
│   ├── decision             "BUY" | "SELL" | "HOLD"
│   ├── conviction           "Strong" | "Moderate" | "Weak"
│   ├── rationale            2-3 sentence explanation
│   ├── time_horizons        {1_day, 1_week, 1_month} decisions
│   └── investment_thesis    list of key thesis points
├── risk_summary
│   ├── risk_score           float 0-10 (10=safe, 0=danger)
│   ├── risk_level           "Critical|High|Moderate|Low|Minimal"
│   ├── primary_risk         single most important risk
│   ├── identified_risks     list of {category, description, severity}
│   └── mitigation_factors   list of strings
├── agent_outputs            per-agent {score, confidence, key_points, reasoning}
├── scores
│   ├── weighted_average     float 0-10
│   └── individual           {earnings, market, risk, sentiment} scores
├── chart_data
│   ├── radar                {labels, scores} for radar chart
│   └── bar                  {agents, scores, weights, weighted_contributions}
└── debate_summary
    ├── timeline             list of per-turn debate entries
    ├── rounds_detail        per-round {trust_evidence, contradictions, decision}
    ├── initial_report       pre-debate analyst summary
    └── final_report         post-debate synthesis paragraph
```

---

## Streamlit Integration

```python
from debate_engine import run_debate

result = run_debate("Apple")  # returns Python dict directly

# Use anywhere in Streamlit:
result["final_decision"]["decision"]        # "BUY"
result["risk_summary"]["risk_score"]        # 6.2
result["scores"]["weighted_average"]        # 7.1
result["chart_data"]["radar"]["scores"]     # [7.2, 6.8, 5.1, 7.5]
result["debate_summary"]["timeline"]        # list of debate turns
```

---

## Notes

- **ChromaDB is local** — no account or cloud service needed. A `chroma_db/` folder is auto-created on first run.
- **First run is slower** (~90 seconds extra) because ChromaDB embeds all chunks. Subsequent runs reuse the cached DB.
- The `debate_output_*.json` files saved to root are for debugging only — the Streamlit UI uses the live Python dict.
- Data is scoped to **Q1 2025** for Apple and Tesla only.

---

## Based On

> Cai et al., *FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis*, arXiv:2509.17395, 2025.