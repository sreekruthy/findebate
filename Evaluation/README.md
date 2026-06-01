# Person 6 — Evaluation Pipeline
## FinDebate: LLM-as-Judge + Cross-Model Benchmark + Statistical Analysis

---

## What Person 6 Must Do (from the task cards)

### Task 1: LLM-as-Judge Pipeline
- Implement GPT-4o-style evaluation using **all 8 scoring dimensions** from Appendix E
- Use a **4-point scale** (1=poor, 4=excellent)  
- Dimensions: readability, linguistic abstractness, coherence, financial key point coverage, background context adequacy, management sentiment conveyance, future outlook analysis, factual accuracy
- Since GPT-4o is paid → we use **Gemini 2.5 Flash** (free) as the judge LLM

### Task 2: Cross-Model Benchmark Runs
- Run **5 LLMs** × **4 conditions** × **15 sampled reports** = 300 evaluations
- Models: GPT-4o, Gemini 2.5 Flash, Llama 4 Maverick, DeepSeek-R1, Claude Sonnet 4
  - Since GPT-4o is paid → use Gemini-1.5-Pro (free) as substitute
- Conditions: Zero-shot, Standard RAG, Multi-agent w/o Debate, FinDebate
- Run **paired t-tests** for statistical significance

---

## What Was Wrong in the Original Code

| File | Problem |
|------|---------|
| `llm_judge.py` | Only 6 dimensions (paper requires 8); wrong JSON key (`recommendations` → should be `investment_recommendations`); no rate limiting; no batching; no resume support |
| `evaluation.py` | Good for basic stats but no LLM scoring, no t-tests |
| Cross-model benchmark | **Completely missing** — not implemented at all |
| Statistical analysis | **Completely missing** — no paired t-tests |
| `llm_judge_results.csv` | Empty (0 rows) — judge never ran successfully |

---

## File Structure

```
evaluation_project/
├── .env                          ← Your API keys go here
├── requirements.txt              ← Python dependencies
├── setup_and_run.sh              ← Master script (submits all SLURM jobs)
│
├── scripts/
│   ├── llm_judge_pipeline.py    ← Task 1: 8-dimension LLM judge (FIXED)
│   ├── cross_model_benchmark.py ← Task 2: 5 models × 4 conditions
│   └── statistical_analysis.py  ← Task 2: Paired t-tests + charts
│
├── slurm_jobs/
│   ├── run_llm_judge.sh         ← SLURM array (2 parallel batches of 32 files)
│   ├── run_cross_model_benchmark.sh ← SLURM single job (5 models run sequentially)
│   └── run_statistics.sh        ← SLURM job (runs after above)
│
├── configs/
│   └── env_template             ← Template for .env
│
├── outputs/                     ← Person 5 p5_output.json files go here
│   ├── *.json                   ← (already done by P5)
│   ├── llm_judge/
│   │   └── llm_judge_results.csv
│   ├── cross_model/
│   │   └── <model>/<condition>/ ← Generated reports
│   └── stats/
│       ├── benchmark_<model>.csv
│       └── final_statistics.json
│
├── charts/                      ← Generated charts
│   ├── main_comparison_table.png
│   ├── dimension_radar.png
│   ├── improvement_bars.png
│   └── ttest_significance.png
│
└── logs/                        ← SLURM output logs
```

---

## Step-by-Step Implementation

### Step 0: Get Free API Keys

**Gemini (required — used as judge AND as one benchmark model):**
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Free tier: 15 requests/minute, 1500/day for `gemini-2.0-flash`
4. **Recommended**: Create a **second key** under the same account for the judge calls (separate RPM counter). Both keys are free.

**NVIDIA NIM (required for Llama 4 Maverick + DeepSeek-R1):**
1. Go to https://build.nvidia.com/
2. Sign in → top-right → "Get API Key"
3. Free tier available for `meta/llama-4-maverick-17b-128e-instruct` and `deepseek-ai/deepseek-r1`

**OpenRouter (required for Claude Sonnet 4 only):**
1. Go to https://openrouter.ai/
2. Create account → "Keys" → "Create Key"
3. Free model: `anthropic/claude-sonnet-4-5:free`

---

### Step 1: Push to GitHub

From your local machine (where you have the zip):

```bash
# Unzip and initialize git
unzip person6_final.zip
cd evaluation_project

# If git not initialized
git init
git remote add origin https://github.com/<your-username>/findebate-eval.git

# Add all files
git add .
git commit -m "Person 6: complete evaluation pipeline with LLM judge, cross-model benchmark, t-tests"
git push -u origin main
```

---

### Step 2: SSH into HPC and Clone

```bash
# SSH into HPC login node
ssh se23ucse176@10.59.121.172

# Clone your repo
git clone https://github.com/<your-username>/findebate-eval.git ~/findebate
cd ~/findebate/evaluation_project
```

---

### Step 3: Set Up Environment

**Option A: Conda (recommended)**
```bash
module load anaconda3 2>/dev/null || true
conda create -n findebate python=3.11 -y
conda activate findebate
pip install -r requirements.txt
```

**Option B: pip venv**
```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r requirements.txt
```

> ⚠️ **Important**: Do NOT install `google-generativeai` (old SDK).  
> Only install `google-genai` (new SDK). They conflict.

---

### Step 4: Configure .env

```bash
cp configs/env_template .env
nano .env
# Fill in:
#   GEMINI_API_KEY=AIza...         ← your primary Gemini key
#   GEMINI_JUDGE_KEY=AIza...       ← optional second Gemini key (recommended)
#   NVIDIA_API_KEY=nvapi-...       ← for Llama 4 + DeepSeek-R1
#   OPENROUTER_API_KEY=sk-or-...   ← for Claude Sonnet 4
```

---

### Step 5: Place Person 5 Outputs

Make sure all `*_p5_output.json` files are in `outputs/`:

```bash
ls outputs/ | grep p5 | wc -l
# Should show ~64 files
```

If you're transferring from another location:
```bash
scp -r /path/to/p5_outputs/*.json se23ucse176@10.59.121.172:~/findebate/evaluation_project/outputs/
```

---

### Step 6: Submit SLURM Jobs (Run in Order)

**⚠️ Important: Run Task 1 first, then Task 2. Do NOT submit them simultaneously.**

```bash
cd ~/findebate/evaluation_project

# ── Task 1: LLM Judge (submit first) ─────────────────────────────────────────
JUDGE_JOB=$(sbatch --parsable slurm_jobs/run_llm_judge.sh)
echo "Judge job submitted: $JUDGE_JOB"
# This runs 2 parallel batches of 32 files each.
# Safe: 2 tasks × 6 calls/min = 12 RPM combined (under 15 RPM free limit).
# Expected time: ~55 minutes.

# ── Task 2: Cross-Model Benchmark (submit after judge finishes) ───────────────
# Option A — submit with automatic dependency (recommended):
BENCH_JOB=$(sbatch --parsable --dependency=afterok:$JUDGE_JOB slurm_jobs/run_cross_model_benchmark.sh)
echo "Benchmark job submitted: $BENCH_JOB (will start after job $JUDGE_JOB)"

# Option B — submit manually once judge shows as COMPLETED in squeue:
#   sbatch slurm_jobs/run_cross_model_benchmark.sh

# ── Task 3: Statistical Analysis (submit after benchmark finishes) ────────────
sbatch --dependency=afterok:$BENCH_JOB slurm_jobs/run_statistics.sh
```

**Monitor jobs:**
```bash
squeue -u se23ucse176
tail -f logs/llm_judge_<JOBID>_0.out
tail -f logs/benchmark_<JOBID>.out
```

---

### Step 7: Run Manually (if SLURM not working)

Request a compute node first:
```bash
srun -N1 --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 --mem=8G --time=06:00:00 --partition=gpu_student --pty /bin/bash
```

Then run sequentially:
```bash
cd ~/findebate/evaluation_project
conda activate findebate

# Task 1: LLM Judge (all 64 files, sequential)
python scripts/llm_judge_pipeline.py

# Task 2a: Cross-model benchmark — one model at a time
python scripts/cross_model_benchmark.py --model gemini_20_flash
python scripts/cross_model_benchmark.py --model gpt4o_equiv
python scripts/cross_model_benchmark.py --model llama4_maverick
python scripts/cross_model_benchmark.py --model deepseek_r1
python scripts/cross_model_benchmark.py --model claude_sonnet4

# Task 2b: Statistical analysis + charts
python scripts/statistical_analysis.py
```

---

### Step 8: Collect Results

After all jobs finish:
```bash
# Check judge results
wc -l outputs/llm_judge/llm_judge_results.csv

# Check benchmark results
for model in gemini_20_flash llama4_maverick deepseek_r1 claude_sonnet4 gpt4o_equiv; do
    echo "$model: $(wc -l < outputs/stats/benchmark_${model}.csv) rows"
done

# View final stats
cat outputs/stats/final_statistics.json | python3 -m json.tool | head -50

# Pull results back to local machine
scp -r se23ucse176@10.59.121.172:~/findebate/evaluation_project/outputs ./
scp -r se23ucse176@10.59.121.172:~/findebate/evaluation_project/charts ./
```

---

## Rate Limits Reference

| API | Model | Free Limit | Safe Sleep | Used for |
|-----|-------|-----------|------------|----------|
| Google AI Studio | gemini-2.0-flash | 15 RPM, 1500 RPD | 10s (judge, 2-task parallel) | LLM judge + generation |
| Google AI Studio | gemini-1.5-pro | 15 RPM | 5s | gpt4o_equiv generation |
| NVIDIA NIM | llama-4-maverick | free tier | 3s | Llama 4 generation |
| NVIDIA NIM | deepseek-r1 | free tier | 3s | DeepSeek-R1 generation |
| OpenRouter | claude-sonnet-4-5:free | 20 RPM | 3s | Claude Sonnet 4 generation |

**Why 2 parallel judge tasks are safe:** Each task sleeps 10s between calls → 6 calls/min per task. Combined = 12 RPM, safely under the 15 RPM free limit.

---

## Output Files Explained

| File | Description |
|------|-------------|
| `outputs/llm_judge/llm_judge_results.csv` | All 64 p5 reports scored on 8 dimensions |
| `outputs/stats/benchmark_<model>.csv` | 15 reports × 4 conditions for one model |
| `outputs/stats/main_table.csv` | Reproduction of paper Table 1 |
| `outputs/stats/final_statistics.json` | Everything: t-tests, averages, breakdown |
| `charts/main_comparison_table.png` | Bar chart by model and condition |
| `charts/dimension_radar.png` | 8-dimension radar chart per condition |
| `charts/improvement_bars.png` | Per-model improvement from FinDebate |
| `charts/ttest_significance.png` | Paired t-test significance visualization |

---

## Troubleshooting

**Gemini quota exhausted (429 error):**
```bash
# Increase sleep in .env:
SLEEP_BETWEEN_CALLS=8.0
SLEEP_GEMINI=8.0
```

**OpenRouter model not found:**
```bash
# Check available free models at: https://openrouter.ai/models?q=free
# Update model_id in cross_model_benchmark.py MODELS dict
```

**Resume after failure:**
All scripts support resume automatically — they check which files are already in the CSV and skip them. Just re-run the same command.

**SLURM job cancelled (time limit):**
```bash
# Re-submit — resume support means it picks up where it left off
sbatch slurm_jobs/run_cross_model_benchmark.sh
```
