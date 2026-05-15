#!/bin/bash
#SBATCH --job-name=p6_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1

# ── USAGE ────────────────────────────────────────────────────────────────────
# Run this AFTER run_llm_judge.sh has fully completed.
# Submit with:
#   sbatch slurm_jobs/run_cross_model_benchmark.sh
#
# Or with dependency on the judge job:
#   sbatch --dependency=afterok:<llm_judge_job_id> slurm_jobs/run_cross_model_benchmark.sh
#
# This runs all 5 models SEQUENTIALLY in a single job to avoid hitting
# free-tier API rate limits (Gemini: 15 RPM, NVIDIA/OpenRouter: shared).
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "SLURM Job ID      : $SLURM_JOB_ID"
echo "Node              : $(hostname)"
echo "Time              : $(date)"
echo "=========================================="

source ~/.bashrc
conda activate findebate 2>/dev/null || source ~/venv/bin/activate 2>/dev/null

cd ~/findebate/evaluation_project || { echo "Project dir not found!"; exit 1; }

mkdir -p logs outputs/cross_model outputs/stats charts

# ── Model list (all 5, run one after another) ─────────────────────────────────
# API routing (set in cross_model_benchmark.py):
#   gemini_20_flash  → Google AI Studio (GEMINI_API_KEY)   — free, 15 RPM
#   gpt4o_equiv      → Google AI Studio (GEMINI_API_KEY)   — gemini-1.5-pro substitute
#   llama4_maverick  → NVIDIA NIM       (NVIDIA_API_KEY)   — free tier
#   deepseek_r1      → NVIDIA NIM       (NVIDIA_API_KEY)   — free tier
#   claude_sonnet4   → OpenRouter       (OPENROUTER_API_KEY) — free tier
MODEL_KEYS=(
    "gemini_20_flash"
    "gpt4o_equiv"
    "llama4_maverick"
    "deepseek_r1"
    "claude_sonnet4"
)

OVERALL_EXIT=0

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo "Starting model: $MODEL_KEY  at $(date)"
    echo "────────────────────────────────────────"

    python scripts/cross_model_benchmark.py --model "$MODEL_KEY"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: $MODEL_KEY exited with code $EXIT_CODE — continuing to next model"
        OVERALL_EXIT=$EXIT_CODE
    else
        echo "Finished $MODEL_KEY successfully at $(date)"
    fi

    # Brief pause between models to let rate-limit windows reset
    sleep 15
done

echo ""
echo "All models complete. Overall exit code: $OVERALL_EXIT at $(date)"
exit $OVERALL_EXIT
