# FinDebate — AI Financial Analysis (MVP)

An AI-powered financial debate system that uses multiple specialized agents to analyze stocks and produce investment recommendations through a structured debate process. Built with Google Gemini, RAG (Retrieval-Augmented Generation), and a Streamlit dashboard.

---

## What It Does

FinDebate runs four analyst AI agents — Earnings, Market, Risk, and Sentiment — each independently analyzing a company using retrieved financial data. The agents then enter a 2-round structured debate (Trust Agent → Skeptic Agent → Leader Agent) to produce a final BUY / SELL / HOLD recommendation with conviction level, time horizons, investment thesis, and risk summary.

Currently supports: **Apple** and **Tesla**

---

## Project Structure

```
findebate/
├── debate_engine.py          # Main pipeline — runs all agents + debate
├── streamlit.py              # Streamlit dashboard UI
├── rag_module.py             # RAG: chunking, embedding, retrieval (ChromaDB)
├── .env                      # API keys (create this yourself)
│
├── Earnings_Analyst/
│   └── earnings_agent.py     # Earnings analyst agent
│
├── Market_Analyst/
│   └── market_agent.py       # Market analyst agent
│
├── risk_analyst/
│   └── risk_agent.py         # Risk analyst agent
│
├── sentiment_analyst/
│   └── sentiment_agent.py    # Sentiment analyst agent
│
└── data/                     # Financial data used by RAG
```

---

## How It Works

```
Financial Data (data/)
        ↓
   RAG Module (ChromaDB + sentence-transformers)
        ↓
  ┌─────────────────────────────────┐
  │  4 Analyst Agents (Gemini LLM)  │
  │  • Earnings   • Market          │
  │  • Risk       • Sentiment       │
  └─────────────────────────────────┘
        ↓
  2-Round Structured Debate
  • Trust Agent    → strengthens thesis
  • Skeptic Agent  → challenges weaknesses
  • Leader Agent   → synthesizes final decision
        ↓
  Final Output: BUY / SELL / HOLD + full analysis
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd findebate
```

### 2. Create your `.env` file

Create a file named `.env` in the root of the project and add your API keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

- **Gemini API key** → get it from [https://ai.google.dev](https://ai.google.dev)
- **Groq API key** → get it from [https://console.groq.com](https://console.groq.com)

> Never commit your `.env` file to GitHub. It's already in `.gitignore`.

### 3. Install dependencies

**Core pipeline dependencies:**
```bash
pip install google-genai groq python-dotenv sentence-transformers chromadb nltk
```

**Streamlit dashboard dependencies:**
```bash
pip install streamlit plotly torchvision torch
```

If you're on Python 3.14 or using a system-managed environment, add `--break-system-packages`:
```bash
pip install streamlit plotly torchvision torch --break-system-packages
```

---

## Running the Analysis

### Option A — Terminal (CLI)

Run the debate engine directly for Apple or Tesla:

```bash
python debate_engine.py Apple
```
```bash
python debate_engine.py Tesla
```

**What you'll see in the terminal:**
```
[FinDebate] Running analyst agents for Apple...
[FinDebate] Starting 2-round debate...
[FinDebate] ✅ Done. Decision: BUY (Strong conviction)

==================================================
  COMPANY:    Apple
  DECISION:   BUY
  CONVICTION: Strong
  RATIONALE:  ...
  RISK LEVEL: Moderate
  W. SCORE:   7.2/10
==================================================
```

**Output file** — a full JSON report is saved automatically:
```
debate_output_apple.json
debate_output_tesla.json
```

This JSON contains everything: all agent scores, debate rounds, investment thesis, risk breakdown, time horizon recommendations, and chart data.

### Option B — Streamlit Dashboard

Launch the visual dashboard:

```bash
streamlit run streamlit.py
```

Then open your browser at: **http://localhost:8501**

The dashboard shows:
- Final decision card (BUY / SELL / HOLD) with conviction
- Risk score gauge
- Individual agent score cards (Earnings, Market, Risk, Sentiment)
- Radar chart + bar chart of agent scores
- Full debate summary timeline (all rounds)
- Time horizon recommendations (1 Day / 1 Week / 1 Month)
- Investment thesis bullet points
- Primary risk summary

---

## API Quota Notes

The free tier of Gemini API (`gemini-2.5-flash`) has a limit of **20 requests per day**. Each full run uses ~10 requests (4 analyst agents + 6 debate agents across 2 rounds), so you get about **2 full runs per day** on the free tier.

**If you hit the quota limit:**
- Wait until midnight (Pacific Time) for the daily reset
- Or switch to `gemini-1.5-flash` in the code — it has 1,500 requests/day free
- Or enable billing at [https://ai.dev](https://ai.dev) (very cheap, ~$0.15/million tokens)

To switch models, change this line in `debate_engine.py` and all agent files:
```python
model="gemini-1.5-flash"  # instead of gemini-2.5-flash
```

---

## Output JSON Structure

The generated `debate_output_<company>.json` contains:

```json
{
  "meta": { "company", "generated_at", "num_rounds" },
  "final_decision": { "decision", "conviction", "rationale", "time_horizons", "investment_thesis" },
  "risk_summary": { "risk_score", "risk_level", "primary_risk", "identified_risks" },
  "agent_outputs": { "earnings", "market", "risk", "sentiment" },
  "scores": { "weighted_average", "individual" },
  "chart_data": { "radar", "bar" },
  "debate_summary": { "timeline", "rounds_detail", "final_report" }
}
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (analysis + debate) | Google Gemini 2.5 Flash |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| Text chunking | NLTK |
| Dashboard | Streamlit + Plotly |
| API calls | `google-genai`, `groq` |
| Config | `python-dotenv` |

---

## Known Limitations (MVP)

- **Only Apple and Tesla** are supported — data is pre-loaded for these two companies
- **Free tier quota** — limited to ~2 runs/day on Gemini free tier
- **Data is static** — financial data is from Q1 2025; not live/real-time
- **No authentication** on the Streamlit app — suitable for local use only
- The debate runs sequentially (not in parallel), so each run takes ~1-2 minutes

---

## MVP Status

This is an **early MVP** built to demonstrate the multi-agent debate architecture. The core pipeline (RAG → 4 agents → 3-agent debate → structured output) is fully functional. Future versions could add more companies, live data feeds, and parallel agent execution.