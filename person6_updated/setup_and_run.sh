#!/bin/bash
# setup_and_run.sh
# ================
# Complete setup script for Person 6 evaluation on HPC (SLURM)
# Run this from the login node: bash setup_and_run.sh
# It sets up the environment and submits all SLURM jobs in the right order.

set -e   # exit on error

echo "=============================================="
echo "  FinDebate Person 6 — Evaluation Setup"
echo "=============================================="

# ── 1. Directory structure ────────────────────────────────────────────────────
echo ""
echo "[1/6] Creating directory structure..."

mkdir -p \
    logs \
    scripts \
    slurm_jobs \
    configs \
    outputs/llm_judge \
    outputs/cross_model \
    outputs/stats \
    charts

echo "  ✅ Directories created"

# ── 2. Python environment ─────────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up Python environment..."

# Option A: conda (preferred on HPC)
if command -v conda &>/dev/null; then
    conda create -n findebate python=3.11 -y 2>/dev/null || true
    conda activate findebate
    echo "  ✅ Conda env 'findebate' activated"
fi

# Install requirements
pip install --quiet --upgrade pip
pip install --quiet \
    python-dotenv \
    pandas \
    numpy \
    scipy \
    matplotlib \
    requests \
    google-genai

echo "  ✅ Python packages installed"

# ── 3. .env file check ───────────────────────────────────────────────────────
echo ""
echo "[3/6] Checking .env configuration..."

if [ ! -f ".env" ]; then
    echo "  ⚠️  No .env file found!"
    echo "  → Copy configs/env_template to .env and fill in your API keys:"
    echo "      cp configs/env_template .env"
    echo "      nano .env"
    echo ""
    echo "  Get free keys at:"
    echo "    Gemini : https://aistudio.google.com/app/apikey"
    echo "    OpenRouter: https://openrouter.ai/"
    echo ""
    echo "  Re-run this script after setting up .env"
    exit 1
fi

# Check keys are set
source .env
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo "  ❌ GEMINI_API_KEY not set in .env!"
    exit 1
fi
if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "your_openrouter_api_key_here" ]; then
    echo "  ⚠️  OPENROUTER_API_KEY not set — cross-model benchmark will only run Gemini models"
fi

echo "  ✅ .env configured"

# ── 4. Verify p5 outputs exist ───────────────────────────────────────────────
echo ""
echo "[4/6] Verifying p5 output files..."

P5_COUNT=$(ls outputs/*.json 2>/dev/null | grep -v final_summary | wc -l)
echo "  Found $P5_COUNT p5 output JSON files in outputs/"

if [ "$P5_COUNT" -lt 15 ]; then
    echo "  ⚠️  Warning: Expected at least 15 files for the benchmark."
    echo "     Make sure Person 5 outputs are in the outputs/ folder."
fi

# ── 5. Submit SLURM jobs ─────────────────────────────────────────────────────
echo ""
echo "[5/6] Submitting SLURM jobs..."

# Job 1: LLM Judge (array of 4 batches)
echo "  → Submitting LLM Judge job array..."
JUDGE_JOB=$(sbatch --parsable slurm_jobs/run_llm_judge.sh)
echo "  ✅ LLM Judge submitted: Job ID $JUDGE_JOB"

# Job 2: Cross-model benchmark (array of 5 models), run in parallel with judge
echo "  → Submitting Cross-Model Benchmark job array..."
BENCH_JOB=$(sbatch --parsable slurm_jobs/run_cross_model_benchmark.sh)
echo "  ✅ Cross-Model Benchmark submitted: Job ID $BENCH_JOB"

# Job 3: Statistical analysis — runs only after BOTH jobs above complete
echo "  → Submitting Statistical Analysis (dependent on jobs $JUDGE_JOB and $BENCH_JOB)..."
STATS_JOB=$(sbatch --parsable \
    --dependency=afterok:${JUDGE_JOB}:${BENCH_JOB} \
    slurm_jobs/run_statistics.sh)
echo "  ✅ Statistical Analysis submitted: Job ID $STATS_JOB"

# ── 6. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Job submission summary:"
echo "  LLM Judge         : Job $JUDGE_JOB  (4 array tasks, ~16 files each)"
echo "  Cross-Model Bench : Job $BENCH_JOB  (5 array tasks, 1 model each)"
echo "  Statistical Anal. : Job $STATS_JOB  (runs after both above finish)"
echo ""
echo "  Monitor with:"
echo "    squeue -u $USER"
echo "    tail -f logs/llm_judge_${JUDGE_JOB}_0.out"
echo "    tail -f logs/benchmark_${BENCH_JOB}_0.out"
echo ""
echo "  Results will appear in:"
echo "    outputs/llm_judge/llm_judge_results.csv"
echo "    outputs/stats/benchmark_<model>.csv"
echo "    outputs/stats/final_statistics.json"
echo "    charts/"
echo ""
echo "=============================================="
echo "  Setup complete! Jobs submitted to SLURM."
echo "=============================================="
