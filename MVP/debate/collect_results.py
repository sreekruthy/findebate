#!/usr/bin/env python3
"""
FinDebate — Person 5
collect_results.py

Run this AFTER all SLURM tasks (or batch) complete.
  1. Counts how many P5 outputs exist
  2. Checks safety-check pass rates
  3. Prints a summary table (stance + positions per file)
  4. Writes p5_outputs/final_summary.json
  5. Lists any missing files so you can re-run just those

Usage:
    python collect_results.py
    python collect_results.py --rerun_missing   # prints re-run commands
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
import debate.configs.config as config


def get_expected_files(p4_dir: str) -> list[str]:
    return sorted([
        f.replace("_p4_output.json", "")
        for f in os.listdir(p4_dir)
        if f.endswith("_p4_output.json")
        and "checkpoint" not in f
        and "batch" not in f
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4_dir",         default=config.P4_OUTPUT_DIR)
    parser.add_argument("--out_dir",        default=config.P5_OUTPUT_DIR)
    parser.add_argument("--rerun_missing",  action="store_true")
    args = parser.parse_args()

    expected = get_expected_files(args.p4_dir)
    out_dir  = Path(args.out_dir)

    rows        = []
    missing     = []
    safety_pass = 0
    safety_fail = 0

    print(f"\n{'='*80}")
    print(f"{'SOURCE FILE':<30} {'ORIG_STANCE':<12} {'FINAL_STANCE':<13} "
          f"{'1D':<6} {'1W':<6} {'1M':<6} {'SAFETY'}")
    print(f"{'='*80}")

    for sf in expected:
        p = out_dir / f"{sf}_p5_output.json"
        if not p.exists():
            missing.append(sf)
            print(f"{sf:<30} {'MISSING':}")
            continue

        with open(p) as f:
            d = json.load(f)

        s = d.get("summary", {})
        orig_stance  = s.get("original_stance",  "?")
        final_stance = s.get("final_stance",      "?")
        d1 = s.get("final_1day",   "?")
        dw = s.get("final_1week",  "?")
        dm = s.get("final_1month", "?")
        safe = "✓" if s.get("safety_passed") else "↩R0"

        if s.get("safety_passed"):
            safety_pass += 1
        else:
            safety_fail += 1

        print(f"{sf:<30} {orig_stance:<12} {final_stance:<13} {d1:<6} {dw:<6} {dm:<6} {safe}")

        rows.append({
            "source_file":   sf,
            "original_stance":  orig_stance,
            "final_stance":     final_stance,
            "final_1day":    d1,
            "final_1week":   dw,
            "final_1month":  dm,
            "safety_passed": s.get("safety_passed"),
        })

    print(f"{'='*80}")
    total    = len(expected)
    complete = len(rows)
    print(f"\nTotal expected : {total}")
    print(f"Completed      : {complete}")
    print(f"Missing        : {len(missing)}")
    print(f"Safety PASSED  : {safety_pass}  ({100*safety_pass/max(complete,1):.1f}%)")
    print(f"Safety FALLBACK: {safety_fail}  ({100*safety_fail/max(complete,1):.1f}%)")

    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for sf in missing:
            print(f"  {sf}")
        if args.rerun_missing:
            print("\nRe-run commands:")
            for sf in missing:
                print(f"  python run_debate.py --source_file {sf}")

    # ── Write final summary ───────────────────────────────────────────────────
    summary = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "total_expected": total,
        "completed":      complete,
        "missing":        missing,
        "safety_pass":    safety_pass,
        "safety_fallback":safety_fail,
        "results":        rows,
    }
    summary_path = out_dir / "final_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
