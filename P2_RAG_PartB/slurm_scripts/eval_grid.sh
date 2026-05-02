#!/bin/bash
#SBATCH --job-name=findebate_eval
#SBATCH --partition=gpu_student
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/dgxa_home/se23ucse176/findebate/logs/eval_%a_%j.log
#SBATCH --error=/dgxa_home/se23ucse176/findebate/logs/eval_%a_%j.err

# All 5 models from the paper
MODELS=(gpt-4o gemini-2.5-flash llama-4-maverick deepseek-r1 claude-sonnet-4)

# Pick model for this job
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Evaluating with model: $MODEL"

source ~/findebate/venv/bin/activate
cd ~/findebate

python3 evaluate.py --model $MODEL

echo "$MODEL evaluation complete!"
