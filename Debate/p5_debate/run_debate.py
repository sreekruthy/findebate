import sys
import os
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import configs.config as config
from src.llm_client  import build_client
from src.algorithm1  import run_safe_debate

def setup_logging(source_file: str, log_dir: str):
    log_path = Path(log_dir) / f"{source_file}_p5.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def already_done(source_file: str, out_dir: str) -> bool:
    out_path = Path(out_dir) / f"{source_file}_p5_output.json"
    return out_path.exists()


def main():
    parser = argparse.ArgumentParser(description="FinDebate Person 5 — single file debate")
    parser.add_argument("--source_file", required=True,
                        help="Base filename without suffix, e.g. ABM_q3_2021")
    parser.add_argument("--p4_dir",  default=config.P4_OUTPUT_DIR)
    parser.add_argument("--p3_dir",  default=config.P3_OUTPUT_DIR)
    parser.add_argument("--out_dir", default=config.P5_OUTPUT_DIR)
    parser.add_argument("--log_dir", default=config.LOG_DIR)
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even if output already exists")
    args = parser.parse_args()

    source_file = args.source_file
    logger = setup_logging(source_file, args.log_dir)

    if not args.force and already_done(source_file, args.out_dir):
        logger.info(f"Output already exists for {source_file} — skipping. Use --force to re-run.")
        sys.exit(0)

    logger.info(f"=== FinDebate P5: {source_file} ===")

    # Load P4 output
    p4_path = Path(args.p4_dir) / f"{source_file}_p4_output.json"
    p4_data = load_json(p4_path)
    if p4_data is None:
        logger.error(f"P4 output not found: {p4_path}")
        sys.exit(1)

    synthesis = p4_data.get("agents", {}).get("synthesis")
    if synthesis is None:
        logger.error(f"No 'synthesis' agent found in {p4_path}")
        sys.exit(1)

    # Inject source_file into synthesis if missing
    if "source_file" not in synthesis:
        synthesis["source_file"] = source_file

    # Load P3 output 
    p3_path = Path(args.p3_dir) / f"{source_file}_p3_output.json"
    p3_data = load_json(p3_path)
    if p3_data is None:
        logger.warning(f"P3 output not found at {p3_path} — proceeding without P3 context.")

    # Build LLM clients
    logger.info("Building LLM clients...")
    trust_client   = build_client("trust",   config)
    skeptic_client = build_client("skeptic", config)
    leader_client  = build_client("leader",  config)

    # Run Algorithm 1
    logger.info("Running Safe Collaborative Debate (Algorithm 1)...")
    optimized, debate_log = run_safe_debate(
        synthesis      = synthesis,
        p3_data        = p3_data,
        p4_data        = p4_data,
        trust_client   = trust_client,
        skeptic_client = skeptic_client,
        leader_client  = leader_client,
    )

    # Build output record 
    output = {
        "source_file":  source_file,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "person":       "P5 — Safe Collaborative Debate",
        "final_source": debate_log.get("final_source", "unknown"),
        "debate_result": optimized,
        "debate_log":   debate_log,
        # Quick-access summary for downstream eval
        "summary": {
            "original_stance":    synthesis.get("overall_stance"),
            "final_stance":       optimized.get("overall_stance",
                                  optimized.get("investment_conclusion", {}).get("final_stance")),
            "original_1day":      synthesis.get("investment_recommendations", {})
                                            .get("one_day",  {}).get("position"),
            "original_1week":     synthesis.get("investment_recommendations", {})
                                            .get("one_week", {}).get("position"),
            "original_1month":    synthesis.get("investment_recommendations", {})
                                            .get("one_month",{}).get("position"),
            "final_1day":         optimized.get("investment_recommendations", {})
                                            .get("one_day",  {}).get("position"),
            "final_1week":        optimized.get("investment_recommendations", {})
                                            .get("one_week", {}).get("position"),
            "final_1month":       optimized.get("investment_recommendations", {})
                                            .get("one_month",{}).get("position"),
            "safety_passed":      debate_log.get("final_source") == "optimized",
        },
    }

    # Write output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_file}_p5_output.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"✓ Saved: {out_path}")
    logger.info(f"  Stance : {output['summary']['original_stance']} → {output['summary']['final_stance']}")
    logger.info(f"  1-day  : {output['summary']['final_1day']}")
    logger.info(f"  Safety : {'PASSED' if output['summary']['safety_passed'] else 'FALLBACK to R0'}")


if __name__ == "__main__":
    main()
