"""
FinDebate — Person 5: Safe Collaborative Debate Mechanism
Configuration file. Edit paths and API keys before running.

PROVIDER CHOICE: Gemini 2.5 Flash (Google AI Studio)
  - Best freely available model (used in the paper itself, Table 1)
  - Free tier: 1,500 requests/day, 1M token context
  - Get your free API key at: https://aistudio.google.com/apikey
  - No credit card required
"""

import os

# ─────────────────────────────────────────────────────────────
# PATHS
# Local:            set to wherever your outputs are
# Supercomputer:    paper project is at findebate/  so:
#                     P4: findebate/outputs
#                     P3: findebate/p3_outputs
#                     P5: findebate/p5_outputs
# ─────────────────────────────────────────────────────────────
P4_OUTPUT_DIR  = "/dgxa_home/se23ucse176/findebate/outputs"      # *_p4_output.json files (Person 4 output)
P3_OUTPUT_DIR  = "/dgxa_home/se23ucse176/findebate/p3_outputs"   # *_p3_output.json files (Person 3 output)
P5_OUTPUT_DIR  = "/dgxa_home/se23ucse176/findebate/p5_outputs"   # Person 5 writes here
LOG_DIR        = "/dgxa_home/se23ucse176/findebate/logs/p5"

# ─────────────────────────────────────────────────────────────
# LLM PROVIDER
# "gemini"    → Gemini 2.5 Flash  (FREE, recommended)
# "openai"    → GPT-4o            (paid)
# "anthropic" → Claude Sonnet 4   (paid)
# ─────────────────────────────────────────────────────────────
DEFAULT_PROVIDER = "gemini"

# ─────────────────────────────────────────────────────────────
# API KEYS  (set as environment variables — never hardcode)
#
# Free Gemini key:  https://aistudio.google.com/apikey
#   export GEMINI_API_KEY="AIza..."
#
# On the cluster, add to ~/.bashrc or your SLURM script:
#   export GEMINI_API_KEY="AIza..."
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",    "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─────────────────────────────────────────────────────────────
# MODEL NAMES
# ─────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-2.5-flash"  # best free model; matches paper Table 1
OPENAI_MODEL    = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-5"

# ─────────────────────────────────────────────────────────────
# GENERATION PARAMETERS  (identical to paper Section 3.1)
# ─────────────────────────────────────────────────────────────
TEMPERATURE       = 0.6
MAX_TOKENS        = 6500
TOP_P             = 0.85
FREQUENCY_PENALTY = 0.1   # used by OpenAI only; ignored elsewhere

# ─────────────────────────────────────────────────────────────
# RETRY / RATE-LIMIT SETTINGS
# Gemini free tier: 15 RPM → delay keeps us safe with SLURM %16 concurrency
# ─────────────────────────────────────────────────────────────
MAX_RETRIES       = 5
RETRY_DELAY_SEC   = 15    # seconds between retries (increase to 30 if hitting 429s)

# ─────────────────────────────────────────────────────────────
# SLURM / HPC SETTINGS
# ─────────────────────────────────────────────────────────────
SLURM_PARTITION   = "gpu"
SLURM_TIME        = "02:00:00"
SLURM_MEM         = "8G"
SLURM_CPUS        = 2
SLURM_ACCOUNT     = ""          # leave blank if not required by your cluster
# Max concurrent SLURM tasks. Gemini free = 15 RPM, each task makes 3 calls,
# so keep at ≤5 concurrent tasks to avoid 429 errors.
SLURM_MAX_CONCURRENT = 5
