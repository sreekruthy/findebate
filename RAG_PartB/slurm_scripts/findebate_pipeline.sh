#!/bin/bash
#SBATCH --job-name=findebate_agents
#SBATCH --partition=gpu_student
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/dgxa_home/se23ucse176/findebate/logs/agent_%a_%j.log
#SBATCH --error=/dgxa_home/se23ucse176/findebate/logs/agent_%a_%j.err

# Array of all 5 agents
AGENTS=(earnings_agent market_agent sentiment_agent valuation_agent risk_agent)

# Pick the agent for this job based on array index
AGENT=${AGENTS[$SLURM_ARRAY_TASK_ID]}

echo "Starting agent: $AGENT"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Activate environment
source ~/findebate/venv/bin/activate

# Move to project directory
cd ~/findebate

# Run the specific agent
python3 agents/$AGENT.py

echo "$AGENT completed!"
