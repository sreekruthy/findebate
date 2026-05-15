#!/bin/bash
#SBATCH --job-name=p6_benchmark
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
#SBATCH --array=0-4          # 5 models, one per task
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1

# ── Setup ────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "SLURM Job ID      : $SLURM_JOB_ID"
echo "Array Task ID     : $SLURM_ARRAY_TASK_ID"
echo "Node              : $(hostname)"
echo "Time              : $(date)"
echo "=========================================="

source ~/.bashrc
conda activate findebate 2>/dev/null || source ~/venv/bin/activate 2>/dev/null

cd ~/findebate/evaluation_project || { echo "Project dir not found!"; exit 1; }

mkdir -p logs outputs/cross_model outputs/stats charts

# ── Model selection by array task ────────────────────────────────────────────
MODEL_KEYS=("gemini_25_flash" "llama4_maverick" "deepseek_r1" "claude_sonnet4" "gpt4o_equiv")
MODEL_KEY=${MODEL_KEYS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: No model key for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running benchmark for model: $MODEL_KEY"

# ── Run benchmark ────────────────────────────────────────────────────────────
python scripts/cross_model_benchmark.py --model "$MODEL_KEY"

EXIT_CODE=$?
echo "Benchmark finished for $MODEL_KEY with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
