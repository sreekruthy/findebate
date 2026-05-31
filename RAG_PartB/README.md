# P2 — RAG Module Part B
**FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis**

**Contributor:** Yajat Kumar  
**Module:** RAG Module — Part B  
**Branch:** `yajat_p2`

---

## Overview

This folder contains the implementation of RAG Module Part B for the FinDebate project, based on the FinDebate research paper (arXiv:2509.17395). My responsibilities covered two tasks:

1. **Multi-level retrieval across all 4 dimensions** from Appendix B of the paper
2. **HPC environment setup** with SLURM job arrays for parallel agent and evaluation execution

This work builds directly on top of P1's RAG infrastructure (ChromaDB + FinLang embeddings).

---

## Folder Structure

```
P2_RAG_PartB/
├── README.md                        ← this file
├── yajat_rag_part2.ipynb            ← main implementation notebook
└── slurm_scripts/
    ├── setup_env.sh                 ← run once to set up HPC environment
    ├── findebate_pipeline.sh        ← SLURM job array for 5 agents in parallel
    └── eval_grid.sh                 ← SLURM job array for 5-model evaluation grid
```

---

## Prerequisites

Before running anything, you need P1's ChromaDB on your Google Drive at:
```
/content/drive/MyDrive/findebate_chromadb
```

This collection should contain **6,963 chunks** from the Earnings2Insights dataset. Verify by running:
```python
collection.count()  # should return 6963
```

---

## How to Run (Google Colab)

**Step 1 — Mount Drive and install packages:**
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install chromadb==0.5.23 sentence-transformers nltk -q
!pip install opentelemetry-api==1.41.1 opentelemetry-sdk==1.41.1 -q
```

**Step 2 — Connect to ChromaDB:**
```python
import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(
    path="/content/drive/MyDrive/findebate_chromadb"
)
collection = chroma_client.get_collection("findebate_rag")
model = SentenceTransformer("FinLang/finance-embeddings-investopedia")

print(f"Connected. Total chunks: {collection.count()}")
```

**Step 3 — Run all cells in `yajat_rag_part2.ipynb`**

---

## What's Implemented

### 4-Dimension Retrieval System (Appendix B)

The `DIMENSION_QUERIES` dictionary implements all 4 dimensions from Appendix B of the paper, with 3 specific queries per dimension:

| Dimension | Focus |
|-----------|-------|
| `general_financial` | Revenue, earnings, guidance, growth, profitability |
| `specialized_metrics` | NIM, ROE, ROA, capital adequacy, NPAs |
| `market_sentiment_risk` | Management tone, risks, challenges, headwinds |
| `multi_query_integration` | Temporal analysis, comparative, longitudinal |

### Agent-to-Dimension Mapping

Each of the 5 specialist agents receives context from specific dimensions:

| Agent | Dimensions Used |
|-------|----------------|
| Earnings Agent | general_financial + specialized_metrics |
| Market Agent | general_financial + multi_query_integration |
| Sentiment Agent | market_sentiment_risk |
| Valuation Agent | specialized_metrics + general_financial |
| Risk Agent | market_sentiment_risk + specialized_metrics |

### Core Functions

```python
# Get context for a specific agent
context = get_agent_context(source_file="ABM_q3_2021", top_k=3)

# Retrieve by specific dimension
results = retrieve_by_dimension("general_financial", top_k=5)

# Retrieve across all 4 dimensions at once
all_dims = retrieve_all_dimensions(top_k=3)

# Base retrieval with optional filter
results = retrieve("revenue growth earnings", top_k=5, doc_type_filter="earnings")
```

---

## How Other Agents Should Use This

Import and call `get_agent_context()` to get the right context for your agent:

```python
# Get context for all 5 agents at once
context = get_agent_context(top_k=3)

earnings_context = context["earnings_agent"]   # list of chunks
market_context   = context["market_agent"]
sentiment_context = context["sentiment_agent"]
valuation_context = context["valuation_agent"]
risk_context     = context["risk_agent"]

# Convert to plain text for your prompt
context_text = "\n".join([r["chunk"] for r in earnings_context])
```

---

## HPC Setup (Mahindra University Supercomputer)

### Login
```bash
ssh se23ucse176@10.59.121.172
```

### First-Time Setup (run once)
```bash
bash ~/findebate/scripts/setup_env.sh
```

### Run All 5 Agents in Parallel
```bash
sbatch ~/findebate/scripts/findebate_pipeline.sh
```
This uses SLURM job arrays (`--array=0-4`) to launch all 5 agents simultaneously on separate compute resources.

### Run 5-Model Evaluation Grid in Parallel
```bash
sbatch ~/findebate/scripts/eval_grid.sh
```
This runs GPT-4o, Gemini 2.5 Flash, Llama 4 Maverick, DeepSeek-R1, and Claude Sonnet 4 simultaneously.

### Monitor Jobs
```bash
squeue -u se23ucse176          # check job status
cat ~/findebate/logs/*.log     # view logs
```

### HPC Specs
- 4 cores (AMD EPYC 7742)
- 16 GB RAM
- GPU: A100 (1g.5gb slice)
- Partition: gpu_student
- Time limit: 6 hours per job

---

## Important Notes

- ChromaDB path on HPC: `~/findebate/findebate_chromadb`
- ChromaDB collection name: `findebate_rag`
- Embedding model: `FinLang/finance-embeddings-investopedia` — do NOT change this
- Total chunks: 6,963 (ECTSum + Professional subset)
- Retrieval is done at **query time** — chunks are not pre-labelled by dimension

---

## Validation Results

All 4 retrieval tests passed on Google Colab:
- Basic retrieval returning chunks with similarity scores ~0.57
- All 4 dimensions returning chunks correctly
- All 5 agents receiving correct context through `get_agent_context()`
- Total chunks confirmed: 6,963
