#!/bin/bash
#SBATCH --job-name=p6_llm_judge
#SBATCH --output=logs/llm_judge_%A_%a.out
#SBATCH --error=logs/llm_judge_%A_%a.err
#SBATCH --array=0-1          # 2 parallel batches, ~32 files each (64 total)
                              # 2 tasks × ~6 calls/min each = ~12 RPM combined,
                              # safely under the gemini-2.0-flash free limit of 15 RPM.
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1

# ── Setup ────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "SLURM Job ID      : $SLURM_JOB_ID"
echo "Array Task ID     : $SLURM_ARRAY_TASK_ID"
echo "Node              : $(hostname)"
echo "Time              : $(date)"
echo "=========================================="

source ~/findebate_env/bin/activate
source ~/findebate_env/bin/activate

cd ~/findebate/person6_updated || { echo "Project dir not found!"; exit 1; }

mkdir -p logs outputs/llm_judge

# ── Batch calculation ────────────────────────────────────────────────────────
# 64 total files split into 2 batches of 32.
# Each task sleeps 10s between calls → 6 calls/min per task.
# Combined across 2 tasks: ~12 RPM — safely under the 15 RPM free tier limit.
TOTAL_FILES=64
ARRAY_SIZE=2
BATCH_SIZE=$(( (TOTAL_FILES + ARRAY_SIZE - 1) / ARRAY_SIZE ))   # = 32

BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( BATCH_START + BATCH_SIZE ))

echo "Processing files $BATCH_START to $BATCH_END (batch size $BATCH_SIZE)"

# ── Run judge ────────────────────────────────────────────────────────────────
# SLEEP_BETWEEN_CALLS=10 → 6 calls/min per task → 12 RPM combined across 2 tasks.
SLEEP_BETWEEN_CALLS=10.0 python scripts/llm_judge_pipeline.py \
    --output-folder outputs \
    --batch-start "$BATCH_START" \
    --batch-end   "$BATCH_END"

EXIT_CODE=$?
echo "Job finished with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
