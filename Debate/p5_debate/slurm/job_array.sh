#!/bin/bash
#SBATCH --job-name=findebate_p5
#SBATCH --partition=cpu_student
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-64%5
# ^^^ %5 = max concurrent tasks.
# Gemini free tier is 15 RPM; each task makes 3 calls,
# so 5 concurrent tasks ≈ 15 RPM — safe.
# Bump to %16 if you upgrade to a paid tier.
# no account specified
#SBATCH --output=/dgxa_home/se23ucse176/findebate/logs/p5/slurm_%A_%a.out
#SBATCH --error=/dgxa_home/se23ucse176/findebate/logs/p5/slurm_%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
source /dgxa_home/se23ucse176/findebate/venv/bin/activate

# ── API Key (FREE — get yours at https://aistudio.google.com/apikey) ─────────
# Add this to ~/.bashrc on the cluster, or uncomment and paste here:
# export GEMINI_API_KEY="AIza..."

# ── Resolve source file from SLURM task ID ────────────────────────────────────
PROJECT_DIR=/dgxa_home/se23ucse176/findebate/p5_debate
FILE_LIST=$PROJECT_DIR/slurm/file_list.txt
SOURCE_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILE_LIST)

echo "=============================="
echo "Task  : $SLURM_ARRAY_TASK_ID / 64"
echo "File  : $SOURCE_FILE"
echo "=============================="

# ── Run ───────────────────────────────────────────────────────────────────────
cd $PROJECT_DIR
python run_debate.py \
    --source_file "$SOURCE_FILE" \
    --p4_dir  "/dgxa_home/se23ucse176/findebate/outputs" \
    --p3_dir  "/dgxa_home/se23ucse176/findebate/p3_outputs" \
    --out_dir "/dgxa_home/se23ucse176/findebate/p5_outputs" \
    --log_dir "/dgxa_home/se23ucse176/findebate/logs/p5"

EXIT_CODE=$?
echo "Finished $SOURCE_FILE with exit code $EXIT_CODE"
exit $EXIT_CODE
