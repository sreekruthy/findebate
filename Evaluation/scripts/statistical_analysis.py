"""
statistical_analysis.py
=======================
Person 6 — Task 2 (Part B): Paired t-tests and Statistical Significance

Runs paired t-tests across conditions (as described in the paper):
  "run all 5 LLMs across all 4 conditions on the 15 sampled reports,
   then run paired t-tests for statistical significance"

Also generates:
  - Summary tables (like Table 1 in the paper)
  - Per-dimension comparison charts
  - Per-model improvement plots
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for HPC
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
STATS_DIR = os.getenv("JUDGE_RESULTS_DIR", "outputs/stats")
CHARTS_DIR = os.getenv("CHARTS_DIR", "charts")
FINAL_OUTPUT = os.getenv("FINAL_STATS_OUTPUT", "outputs/stats/final_statistics.json")

CONDITIONS = ["zero_shot", "standard_rag", "multi_agent", "findebate"]
CONDITION_LABELS = {
    "zero_shot": "Zero-shot",
    "standard_rag": "Standard RAG",
    "multi_agent": "Multi-agent w/o Debate",
    "findebate": "FinDebate"
}

MODEL_KEYS = ["gemini_20_flash", "llama4_maverick", "deepseek_r1", "claude_sonnet4", "gpt4o_equiv"]
MODEL_DISPLAY = {
    "gemini_20_flash": "Gemini 2.5 Flash",
    "llama4_maverick": "Llama 4 Maverick",
    "deepseek_r1": "DeepSeek-R1",
    "claude_sonnet4": "Claude Sonnet 4",
    "gpt4o_equiv": "GPT-4o / Gemini-Pro"
}

DIMENSIONS = [
    "readability", "linguistic_abstractness", "coherence",
    "financial_key_point_coverage", "background_context_adequacy",
    "management_sentiment_conveyance", "future_outlook_analysis", "factual_accuracy"
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_benchmark_results() -> pd.DataFrame:
    """Load and merge all per-model benchmark CSVs."""
    all_dfs = []

    for model_key in MODEL_KEYS:
        csv_path = os.path.join(STATS_DIR, f"benchmark_{model_key}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df["model"] = model_key
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {csv_path}")
            except Exception as e:
                logger.warning(f"Could not load {csv_path}: {e}")
        else:
            logger.warning(f"Missing benchmark file: {csv_path}")

    if not all_dfs:
        logger.error("No benchmark results found! Run cross_model_benchmark.py first.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Remove failed rows
    df = df[df["avg_overall"] > 0].copy()
    logger.info(f"Total valid rows: {len(df)}")
    return df


# ── Main Table (Paper Table 1) ────────────────────────────────────────────────

def generate_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Table 1 from the paper:
    Base Model | Zero-shot | Standard RAG | Multi-agent w/o Debate | FinDebate | Overall Improvement
    """
    rows = []
    for model_key in MODEL_KEYS:
        mdf = df[df["model"] == model_key]
        if mdf.empty:
            continue

        row = {"Base Model": MODEL_DISPLAY.get(model_key, model_key)}
        scores_by_condition = {}

        for cond in CONDITIONS:
            cdf = mdf[mdf["condition"] == cond]
            if not cdf.empty:
                score = cdf["avg_overall"].mean()
            else:
                score = float("nan")
            row[CONDITION_LABELS[cond]] = round(score, 2)
            scores_by_condition[cond] = score

        # Overall improvement: findebate - zero_shot
        zs = scores_by_condition.get("zero_shot", float("nan"))
        fd = scores_by_condition.get("findebate", float("nan"))
        row["Overall Improvement"] = round(fd - zs, 2) if not (np.isnan(zs) or np.isnan(fd)) else float("nan")

        rows.append(row)

    table = pd.DataFrame(rows)
    return table


# ── Paired t-tests ────────────────────────────────────────────────────────────

def run_paired_ttests(df: pd.DataFrame) -> dict:
    """
    Run paired t-tests between FinDebate and each baseline condition.
    Paired = same (model, file) pairs.
    Returns dict of {comparison: {t_stat, p_value, significant, n_pairs}}
    """
    results = {}
    comparisons = [
        ("findebate", "zero_shot"),
        ("findebate", "standard_rag"),
        ("findebate", "multi_agent"),
        ("multi_agent", "zero_shot"),
        ("multi_agent", "standard_rag"),
        ("standard_rag", "zero_shot"),
    ]

    for cond_a, cond_b in comparisons:
        pair_scores_a = []
        pair_scores_b = []

        for model_key in MODEL_KEYS:
            mdf = df[df["model"] == model_key]
            files_a = mdf[mdf["condition"] == cond_a][["file", "avg_overall"]].set_index("file")
            files_b = mdf[mdf["condition"] == cond_b][["file", "avg_overall"]].set_index("file")

            common_files = files_a.index.intersection(files_b.index)
            for f in common_files:
                pair_scores_a.append(files_a.loc[f, "avg_overall"])
                pair_scores_b.append(files_b.loc[f, "avg_overall"])

        if len(pair_scores_a) < 2:
            logger.warning(f"Not enough pairs for {cond_a} vs {cond_b} ({len(pair_scores_a)} pairs)")
            continue

        a = np.array(pair_scores_a)
        b = np.array(pair_scores_b)

        t_stat, p_val = stats.ttest_rel(a, b)
        label = f"{CONDITION_LABELS[cond_a]} vs {CONDITION_LABELS[cond_b]}"

        results[label] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_001": bool(p_val < 0.001),
            "significant_005": bool(p_val < 0.05),
            "n_pairs": len(pair_scores_a),
            "mean_a": round(float(a.mean()), 4),
            "mean_b": round(float(b.mean()), 4),
            "mean_diff": round(float((a - b).mean()), 4),
        }

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        logger.info(f"{label}: t={t_stat:.3f}, p={p_val:.4f} {sig} (n={len(pair_scores_a)})")

    return results


# ── Per-Dimension Analysis ────────────────────────────────────────────────────

def dimension_breakdown(df: pd.DataFrame) -> dict:
    """Mean score per dimension per condition, averaged across all models."""
    breakdown = {}
    for cond in CONDITIONS:
        cdf = df[df["condition"] == cond]
        breakdown[cond] = {
            dim: round(cdf[dim].mean(), 3)
            for dim in DIMENSIONS
            if dim in cdf.columns and not cdf[dim].isna().all()
        }
    return breakdown


# ── Chart Generation ──────────────────────────────────────────────────────────

def plot_main_table(table: pd.DataFrame, save_dir: str):
    """Bar chart: FinDebate vs baselines per model."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(table))
    width = 0.2
    cond_cols = [CONDITION_LABELS[c] for c in CONDITIONS]
    colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]

    for i, (cond, color) in enumerate(zip(cond_cols, colors)):
        if cond in table.columns:
            vals = table[cond].fillna(0).values
            ax.bar(x + i * width, vals, width, label=cond, color=color)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(table["Base Model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Avg Score (1-4 scale)")
    ax.set_title("FinDebate Performance Comparison Across Models and Conditions")
    ax.legend(loc="lower right")
    ax.set_ylim(2.5, 4.0)
    ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (3.0)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "main_comparison_table.png"), dpi=150)
    plt.close()
    logger.info("Saved: main_comparison_table.png")


def plot_dimension_radar(breakdown: dict, save_dir: str):
    """Radar/spider chart of dimension scores per condition."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    dims = DIMENSIONS
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    short_labels = [d.replace("_", "\n") for d in dims]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {"zero_shot": "blue", "standard_rag": "orange",
              "multi_agent": "green", "findebate": "red"}

    for cond, scores in breakdown.items():
        values = [scores.get(d, 0) for d in dims]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=CONDITION_LABELS.get(cond, cond),
                color=colors.get(cond, "gray"))
        ax.fill(angles, values, alpha=0.1, color=colors.get(cond, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, size=8)
    ax.set_ylim(1, 4)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_title("8-Dimension Evaluation by Condition", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dimension_radar.png"), dpi=150)
    plt.close()
    logger.info("Saved: dimension_radar.png")


def plot_improvement_bars(table: pd.DataFrame, save_dir: str):
    """Bar chart of Overall Improvement per model."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if "Overall Improvement" not in table.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(
        table["Base Model"],
        table["Overall Improvement"].fillna(0),
        color=colors[:len(table)]
    )
    ax.set_ylabel("Score Improvement (FinDebate − Zero-shot)")
    ax.set_title("Overall Improvement from FinDebate Framework")
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, table["Overall Improvement"].fillna(0)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"+{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "improvement_bars.png"), dpi=150)
    plt.close()
    logger.info("Saved: improvement_bars.png")


def plot_ttest_significance(ttest_results: dict, save_dir: str):
    """Heatmap-style bar chart of p-values for paired t-tests."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    labels = list(ttest_results.keys())
    pvals = [ttest_results[l]["p_value"] for l in labels]
    t_stats = [ttest_results[l]["t_statistic"] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # T-statistics
    colors = ["green" if t > 0 else "red" for t in t_stats]
    ax1.barh(labels, t_stats, color=colors, alpha=0.7)
    ax1.axvline(x=0, color="black", linewidth=0.8)
    ax1.set_title("Paired t-test: t-statistics")
    ax1.set_xlabel("t-statistic")

    # P-values with significance markers
    bar_colors = ["#2ca02c" if p < 0.001 else "#ff7f0e" if p < 0.05 else "#d62728" for p in pvals]
    ax2.barh(labels, [-np.log10(max(p, 1e-10)) for p in pvals], color=bar_colors, alpha=0.8)
    ax2.axvline(x=-np.log10(0.05), color="orange", linestyle="--", label="p=0.05")
    ax2.axvline(x=-np.log10(0.001), color="green", linestyle="--", label="p=0.001")
    ax2.set_title("Paired t-test: -log10(p-value)")
    ax2.set_xlabel("-log10(p-value)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ttest_significance.png"), dpi=150)
    plt.close()
    logger.info("Saved: ttest_significance.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis():
    Path(STATS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHARTS_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("Loading benchmark results...")
    df = load_all_benchmark_results()

    if df.empty:
        logger.error("No data to analyze. Exiting.")
        return

    # 1. Main performance table
    logger.info("\n── MAIN PERFORMANCE TABLE ──")
    table = generate_main_table(df)
    print("\n" + table.to_string(index=False))
    table.to_csv(os.path.join(STATS_DIR, "main_table.csv"), index=False)
    plot_main_table(table, CHARTS_DIR)

    # 2. Paired t-tests
    logger.info("\n── PAIRED T-TESTS ──")
    ttest_results = run_paired_ttests(df)
    plot_ttest_significance(ttest_results, CHARTS_DIR)

    # 3. Per-dimension breakdown
    logger.info("\n── DIMENSION BREAKDOWN ──")
    breakdown = dimension_breakdown(df)
    plot_dimension_radar(breakdown, CHARTS_DIR)

    # 4. Improvement bars
    plot_improvement_bars(table, CHARTS_DIR)

    # 5. Save full JSON summary
    final = {
        "n_models": len(df["model"].unique()),
        "n_conditions": len(df["condition"].unique()),
        "n_files": len(df["file"].unique()),
        "n_total_rows": len(df),
        "main_table": table.to_dict(orient="records"),
        "paired_ttests": ttest_results,
        "dimension_breakdown": breakdown,
        "findebate_avg_overall": round(df[df["condition"] == "findebate"]["avg_overall"].mean(), 4),
        "zero_shot_avg_overall": round(df[df["condition"] == "zero_shot"]["avg_overall"].mean(), 4),
    }

    with open(FINAL_OUTPUT, "w") as f:
        json.dump(final, f, indent=2)

    logger.info(f"\nFull statistics saved to: {FINAL_OUTPUT}")
    logger.info("\n── SUMMARY ──")
    logger.info(f"  FinDebate avg overall score: {final['findebate_avg_overall']:.3f}")
    logger.info(f"  Zero-shot avg overall score: {final['zero_shot_avg_overall']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", default=STATS_DIR)
    parser.add_argument("--charts-dir", default=CHARTS_DIR)
    args = parser.parse_args()
    STATS_DIR = args.stats_dir
    CHARTS_DIR = args.charts_dir
    run_analysis()
