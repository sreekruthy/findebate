#!/usr/bin/env python3
"""
FinDebate — Person 5
SLURM job array generator.

Run this script ONCE on the login node to generate two files:
  1. slurm/job_array.sh     — the SLURM submission script
  2. slurm/file_list.txt    — one source_file name per line (indexed by $SLURM_ARRAY_TASK_ID)

Then submit with:
    sbatch slurm/job_array.sh

Each SLURM task processes exactly one file → all 64 run in parallel.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import debate.configs.config as config


def get_all_source_files(p4_dir: str) -> list[str]:
    return sorted([
        f.replace("_p4_output.json", "")
        for f in os.listdir(p4_dir)
        if f.endswith("_p4_output.json")
        and "checkpoint" not in f
        and "batch" not in f
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4_dir",    default=config.P4_OUTPUT_DIR)
    parser.add_argument("--p3_dir",    default=config.P3_OUTPUT_DIR)
    parser.add_argument("--out_dir",   default=config.P5_OUTPUT_DIR)
    parser.add_argument("--log_dir",   default=config.LOG_DIR)
    parser.add_argument("--project_dir", default="$HOME/findebate/p5_debate",
                        help="Absolute path to this project on the cluster")
    parser.add_argument("--conda_env", default="findebate",
                        help="Conda environment name (must have openai/anthropic installed)")
    parser.add_argument("--partition", default=config.SLURM_PARTITION)
    parser.add_argument("--time",      default=config.SLURM_TIME)
    parser.add_argument("--mem",       default=config.SLURM_MEM)
    parser.add_argument("--cpus",      default=config.SLURM_CPUS)
    parser.add_argument("--account",   default=config.SLURM_ACCOUNT)
    args = parser.parse_args()

    files = get_all_source_files(args.p4_dir)
    n = len(files)
    print(f"Found {n} files.")

    # ── Write file list ───────────────────────────────────────────────────────
    slurm_dir = Path("slurm")
    slurm_dir.mkdir(exist_ok=True)

    file_list_path = slurm_dir / "file_list.txt"
    with open(file_list_path, "w") as f:
        for sf in files:
            f.write(sf + "\n")
    print(f"Written: {file_list_path}  ({n} entries)")

    # ── Write SLURM script ────────────────────────────────────────────────────
    account_line = f"#SBATCH --account={args.account}" if args.account else "# no account specified"

    script = f"""#!/bin/bash
#SBATCH --job-name=findebate_p5
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --array=1-{n}%{config.SLURM_MAX_CONCURRENT}
# ^^^ %{config.SLURM_MAX_CONCURRENT} = max concurrent tasks.
# Gemini free tier is 15 RPM; each task makes 3 calls,
# so {config.SLURM_MAX_CONCURRENT} concurrent tasks ≈ {config.SLURM_MAX_CONCURRENT * 3} RPM — safe.
# Bump to %16 if you upgrade to a paid tier.
{account_line}
#SBATCH --output={args.log_dir}/slurm_%A_%a.out
#SBATCH --error={args.log_dir}/slurm_%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
source $HOME/miniconda3/etc/profile.d/conda.sh   # adjust path if needed
conda activate {args.conda_env}

# ── API Key (FREE — get yours at https://aistudio.google.com/apikey) ─────────
# Add this to ~/.bashrc on the cluster, or uncomment and paste here:
# export GEMINI_API_KEY="AIza..."

# ── Resolve source file from SLURM task ID ────────────────────────────────────
PROJECT_DIR={args.project_dir}
FILE_LIST=$PROJECT_DIR/slurm/file_list.txt
SOURCE_FILE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" $FILE_LIST)

echo "=============================="
echo "Task  : $SLURM_ARRAY_TASK_ID / {n}"
echo "File  : $SOURCE_FILE"
echo "=============================="

# ── Run ───────────────────────────────────────────────────────────────────────
cd $PROJECT_DIR
python run_debate.py \\
    --source_file "$SOURCE_FILE" \\
    --p4_dir  "{args.p4_dir}" \\
    --p3_dir  "{args.p3_dir}" \\
    --out_dir "{args.out_dir}" \\
    --log_dir "{args.log_dir}"

EXIT_CODE=$?
echo "Finished $SOURCE_FILE with exit code $EXIT_CODE"
exit $EXIT_CODE
"""

    slurm_script_path = slurm_dir / "job_array.sh"
    with open(slurm_script_path, "w") as f:
        f.write(script)
    slurm_script_path.chmod(0o755)
    print(f"Written: {slurm_script_path}")
    print()
    print("Next steps on the cluster:")
    print(f"  1. Copy this directory to {args.project_dir}")
    print(f"  2. Set OPENAI_API_KEY / ANTHROPIC_API_KEY in your environment")
    print(f"  3. sbatch slurm/job_array.sh")
    print(f"  4. squeue -u $USER  (to monitor)")
    print(f"  5. python collect_results.py  (after all tasks finish)")


if __name__ == "__main__":
    main()
