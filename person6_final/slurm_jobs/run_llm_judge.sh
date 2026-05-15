#!/bin/bash
#SBATCH --job-name=p6_llm_judge
#SBATCH --output=logs/llm_judge_%A_%a.out
#SBATCH --error=logs/llm_judge_%A_%a.err
#SBATCH --array=0-3          # 4 batches, ~16 files each (64 total p5 outputs)
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1

# ── Setup ────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "SLURM Job ID      : $SLURM_JOB_ID"
echo "Array Task ID     : $SLURM_ARRAY_TASK_ID"
echo "Node              : $(hostname)"
echo "Time              : $(date)"
echo "=========================================="

# Activate your conda/venv environment
# Replace 'findebate' with your actual env name
source ~/.bashrc
conda activate findebate 2>/dev/null || source ~/venv/bin/activate 2>/dev/null

# Navigate to project directory (update this path)
cd ~/findebate/evaluation_project || { echo "Project dir not found!"; exit 1; }

mkdir -p logs outputs/llm_judge

# ── Batch calculation ────────────────────────────────────────────────────────
# Total files ≈ 64. Split into 4 batches of 16.
TOTAL_FILES=64
ARRAY_SIZE=4
BATCH_SIZE=$(( (TOTAL_FILES + ARRAY_SIZE - 1) / ARRAY_SIZE ))

BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( BATCH_START + BATCH_SIZE ))

echo "Processing files $BATCH_START to $BATCH_END"

# ── Run judge ────────────────────────────────────────────────────────────────
python scripts/llm_judge_pipeline.py \
    --output-folder outputs \
    --batch-start "$BATCH_START" \
    --batch-end "$BATCH_END"

EXIT_CODE=$?
echo "Job finished with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
