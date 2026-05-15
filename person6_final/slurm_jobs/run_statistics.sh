#!/bin/bash
#SBATCH --job-name=p6_stats
#SBATCH --output=logs/stats_%j.out
#SBATCH --error=logs/stats_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1

# Run AFTER run_llm_judge.sh and run_cross_model_benchmark.sh are DONE
# Usage: sbatch --dependency=afterok:<judge_job_id>:<benchmark_job_id> run_statistics.sh

echo "=========================================="
echo "SLURM Job ID      : $SLURM_JOB_ID"
echo "Node              : $(hostname)"
echo "Time              : $(date)"
echo "=========================================="

source ~/.bashrc
conda activate findebate 2>/dev/null || source ~/venv/bin/activate 2>/dev/null

cd ~/findebate/evaluation_project || { echo "Project dir not found!"; exit 1; }

mkdir -p logs charts outputs/stats

python scripts/statistical_analysis.py \
    --stats-dir outputs/stats \
    --charts-dir charts

EXIT_CODE=$?
echo "Statistical analysis finished with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
