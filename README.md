# FinDebate: Multi-Agent Financial Analysis Pipeline

FinDebate is an end-to-end multi-agent financial analysis system that uses retrieval-augmented generation, specialist analyst agents, report synthesis, debate-based refinement, and LLM-as-judge evaluation to generate investment-oriented company reports from earnings call transcripts.

The project implements a full pipeline across:

1. RAG retrieval over financial transcript chunks
2. Earnings, market, and sentiment analysis agents
3. Valuation and risk analysis agents
4. Report synthesis
5. Trust–Skeptic–Leader debate refinement
6. LLM-based evaluation
7. A web portal for precomputed and live analysis

---

## Project Structure

```text
findebate/
├── RAG/
│   └── Initial RAG implementation and transcript chunking
│
├── RAG_PartB/
│   ├── Multi-dimensional retrieval system
│   ├── ChromaDB integration
│   └── SLURM/HPC scripts
│
├── Market + Sentiment + Earnings/
│   ├── Earnings_Analyst/
│   ├── Market_Analyst/
│   ├── sentiment_analyst/
│   ├── run_p3_agents.py
│   └── p3_outputs/
│
├── Valuation + Risk + Report Synthesis/
│   ├── Pipeline.py
│   ├── P4_Analyst.ipynb
│   └── P4_outputs/
│
├── Debate/
│   ├── src/
│   ├── configs/
│   ├── run_debate.py
│   ├── run_batch.py
│   ├── collect_results.py
│   └── p5_outputs/
│
├── Evaluation/
│   ├── LLM judge evaluation
│   ├── Cross-model benchmark outputs
│   └── evaluation logs/results
│
└── findebate-portal/
    ├── backend/
    ├── frontend/
    └── start.sh
```

---

## Pipeline Overview

```text
Earnings Call Transcript
        ↓
RAG Retrieval
        ↓
P3 Agents
  - Earnings Analyst
  - Market Analyst
  - Sentiment Analyst
        ↓
P4 Agents
  - Valuation Analyst
  - Risk Analyst
  - Report Synthesizer
        ↓
P5 Debate
  - Trust Agent
  - Skeptic Agent
  - Leader Agent
        ↓
P6 Evaluation
  - LLM Judge
  - Cross-model benchmarks
        ↓
Final Investment Report
```

---

## Modules

### 1. RAG Retrieval

The RAG layer retrieves relevant transcript chunks from a ChromaDB vector store.

It supports multi-dimensional retrieval across:

- General financial performance
- Specialized financial metrics
- Market sentiment and risk
- Multi-query temporal integration

The retrieval system is used by all downstream analyst agents.

---

### 2. Market, Earnings, and Sentiment Agents

Located in:

```text
Market + Sentiment + Earnings/
```

This stage runs three specialist agents:

- **Earnings Analyst**: revenue, profitability, earnings quality, guidance
- **Market Analyst**: 1-day, 1-week, and 1-month market outlook
- **Sentiment Analyst**: management tone, investor psychology, behavioral signals

Run:

```bash
cd "Market + Sentiment + Earnings"
python run_p3_agents.py
```

Outputs are written to:

```text
Market + Sentiment + Earnings/p3_outputs/
```

---

### 3. Valuation, Risk, and Report Synthesis

Located in:

```text
Valuation + Risk + Report Synthesis/
```

This stage runs:

- **Valuation Analyst**
- **Risk Analyst**
- **Report Synthesizer**

It combines P3 outputs with valuation and risk analysis to create a structured institutional-style investment report.

Run:

```bash
cd "Valuation + Risk + Report Synthesis"
python Pipeline.py
```

Outputs are written to:

```text
Valuation + Risk + Report Synthesis/P4_outputs/
```

---

### 4. Debate Refinement

Located in:

```text
Debate/
```

The debate mechanism uses three agents:

- **Trust Agent**: strengthens supporting evidence
- **Skeptic Agent**: adds risks and counterarguments
- **Leader Agent**: synthesizes the final refined report

Run one file:

```bash
cd Debate
python run_debate.py
```

Run batch processing:

```bash
python run_batch.py
```

Collect final results:

```bash
python collect_results.py
```

Outputs are written to:

```text
Debate/p5_outputs/
```

---

### 5. Evaluation

Located in:

```text
Evaluation/
```

This module evaluates final reports using LLM-as-judge scoring and cross-model benchmarking.

Evaluation dimensions include:

- Readability
- Coherence
- Financial key point coverage
- Background context adequacy
- Management sentiment conveyance
- Future outlook analysis
- Factual accuracy

---

## Web Portal

Located in:

```text
findebate-portal/
```

The portal provides a full-stack interface for viewing precomputed reports and running the live pipeline.

### Run the Portal

```bash
cd findebate-portal
bash start.sh
```

Backend:

```text
http://localhost:8000
```

Frontend:

```text
http://localhost:5174
```

### Portal Modes

- **Precomputed Mode**: Uses existing debate output JSON files.
- **Live Mode**: Runs RAG, Agents, Debate, and Evaluation dynamically.

Example live input:

```text
Analyze ABM revenue guidance, earnings quality, valuation, sentiment, and key risks.
```

For best live results, ask about companies available in the transcript dataset, such as:

```text
ABM, CMI, DE, PCAR, UNH, WYNN
```

---

## Environment Variables

Create a `.env` file locally with:

```bash
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
NVIDIA_API_KEY=your_nvidia_key
```

Optional provider settings:

```bash
FINDEBATE_P4_PROVIDER=groq
FINDEBATE_P5_PROVIDER=groq
FINDEBATE_RETRIEVAL_MODE=sqlite
```

Do not commit `.env` files or API keys to GitHub.

---

## Installation

Install common Python dependencies:

```bash
pip install python-dotenv chromadb groq google-genai sentence-transformers nltk fastapi uvicorn sse-starlette pydantic
```

For the portal frontend:

```bash
cd findebate-portal/frontend
npm install
```

---

## Data and Outputs

The pipeline uses earnings call transcript chunks stored in ChromaDB.

Common output folders:

```text
Market + Sentiment + Earnings/p3_outputs/
Valuation + Risk + Report Synthesis/P4_outputs/
Debate/p5_outputs/
Evaluation/outputs/
findebate-portal/backend/data/p5_outputs/
```

Large generated files, logs, `.env` files, and local databases should not be committed unless intentionally included for reproducibility.

---

## Example End-to-End Workflow

```bash
# 1. Run P3(market,sentiment,earnings) agents
cd "Market + Sentiment + Earnings"
python run_p3_agents.py

# 2. Run P4 valuation, risk, and synthesis
cd "../Valuation + Risk + Report Synthesis"
python Pipeline.py

# 3. Run P5 debate refinement
cd "../Debate"
python run_batch.py
python collect_results.py

# 4. Launch web portal
cd "../findebate-portal"
bash start.sh
```

---

## Key Features

- Multi-agent financial analysis pipeline
- Retrieval-augmented generation over earnings transcripts
- Specialist analyst roles for earnings, market, sentiment, valuation, and risk
- Debate-based report refinement with safety checks
- LLM judge evaluation and benchmark support
- Web portal for precomputed and live analysis
- Supports Groq, Gemini, and NVIDIA API keys

---

## Notes

- Live mode depends on local ChromaDB files, API keys, and agent folders being present.
- Precomputed mode in the portal can run immediately if P5 output JSON files are available.
- For faster live execution, Groq is recommended for P4 and P5 providers.
- HuggingFace-based embedding retrieval can be slower; SQLite keyword retrieval is useful for quick local demos.

---

