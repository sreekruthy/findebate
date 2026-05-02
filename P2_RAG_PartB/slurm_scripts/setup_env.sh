#!/bin/bash
# setup_env.sh
# Run this ONCE on HPC to install all dependencies

echo "Setting up FinDebate environment..."

# Load Python module
module load python/3.10

# Create virtual environment
python3 -m venv ~/findebate/venv
source ~/findebate/venv/bin/activate

# Install all required packages
pip install chromadb==0.5.23
pip install sentence-transformers
pip install nltk
pip install google-generativeai
pip install opentelemetry-api==1.41.1
pip install opentelemetry-sdk==1.41.1

echo "Environment setup complete!"
