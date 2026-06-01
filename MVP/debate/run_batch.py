#!/usr/bin/env python3
"""
FinDebate — Person 5
Batch runner — processes all P4 files sequentially (local machine).
For parallel HPC runs, use the SLURM scripts instead.

Usage:
    python run_batch.py
    python run_batch.py --force          # re-run everything
    python run_batch.py --max_files 5    # test on first 5 files
"""

import sys
import os
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
import MVP.debate.configs.config as config


def get_all_source_files(p4_dir: str) -> list[str]:
    return sorted([
        f.replace("_p4_output.json", "")
        for f in os.listdir(p4_dir)
        if f.endswith("_p4_output.json")
        and "checkpoint" not in f
        and "batch" not in f
    ])


def already_done(source_file: str, out_dir: str) -> bool:
    return (Path(out_dir) / f"{source_file}_p5_output.json").exists()


def main():
    parser = argparse.ArgumentParser(description="FinDebate P5 — batch run")
    parser.add_argument("--p4_dir",    default=config.P4_OUTPUT_DIR)
    parser.add_argument("--p3_dir",    default=config.P3_OUTPUT_DIR)
    parser.add_argument("--out_dir",   default=config.P5_OUTPUT_DIR)
    parser.add_argument("--log_dir",   default=config.LOG_DIR)
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Process only first N files (for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    files = get_all_source_files(args.p4_dir)
    if args.max_files:
        files = files[:args.max_files]

    total = len(files)
    logger.info(f"Found {total} files to process.")

    results = []
    for i, sf in enumerate(files, 1):
        if not args.force and already_done(sf, args.out_dir):
            logger.info(f"[{i}/{total}] SKIP (exists): {sf}")
            results.append({"source_file": sf, "status": "skipped"})
            continue

        logger.info(f"[{i}/{total}] Processing: {sf}")
        cmd = [
            sys.executable, "run_debate.py",
            "--source_file", sf,
            "--p4_dir",  args.p4_dir,
            "--p3_dir",  args.p3_dir,
            "--out_dir", args.out_dir,
            "--log_dir", args.log_dir,
        ]
        if args.force:
            cmd.append("--force")

        ret = subprocess.run(cmd, capture_output=False)
        status = "success" if ret.returncode == 0 else f"error (code {ret.returncode})"
        results.append({"source_file": sf, "status": status})
        logger.info(f"[{i}/{total}] {sf}: {status}")

    # ── Write batch summary ───────────────────────────────────────────────────
    summary = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "total":      total,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "skipped":    sum(1 for r in results if r["status"] == "skipped"),
        "failed":     sum(1 for r in results if "error" in r["status"]),
        "results":    results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"BATCH COMPLETE: {summary['successful']} success | "
                f"{summary['skipped']} skipped | {summary['failed']} failed")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
