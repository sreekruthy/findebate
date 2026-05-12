"""
FinDebate — Person 5
Algorithm 1: Safe Collaborative Debate
Implements has_recommendations() and core_compromised() safety checks.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Valid position tokens
VALID_POSITIONS = {"LONG", "SHORT", "NEUTRAL"}
VALID_STANCES   = {"BULLISH", "BEARISH", "NEUTRAL"}


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY CHECK 1: has_recommendations(R0)
# "if ¬has_recommendations(R0) then return R0"
# ─────────────────────────────────────────────────────────────────────────────

def has_recommendations(synthesis: dict) -> bool:
    """
    Returns True only if the original report contains valid investment
    recommendations for all three timeframes and a recognizable stance.
    Any missing or invalid field causes an early return of R0 (the original).
    """
    # Must have overall stance
    stance = synthesis.get("overall_stance", "").upper().strip()
    if stance not in VALID_STANCES:
        logger.warning(f"has_recommendations FAILED: overall_stance={stance!r}")
        return False

    rec = synthesis.get("investment_recommendations", {})
    for horizon in ("one_day", "one_week", "one_month"):
        h = rec.get(horizon, {})
        pos = h.get("position", "").upper().strip()
        if pos not in VALID_POSITIONS:
            logger.warning(f"has_recommendations FAILED: {horizon}.position={pos!r}")
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY CHECK 2: core_compromised(R*, R0)
# "if core_compromised(R*, R0) then return R0"
# ─────────────────────────────────────────────────────────────────────────────

def core_compromised(optimized: dict, original: dict) -> bool:
    """
    Returns True if the debate output has CHANGED any core recommendation
    compared to the original synthesis.  If True, Algorithm 1 discards R*
    and returns R0 unchanged.

    Checks:
      • overall_stance
      • investment_recommendations.*.position for each timeframe
    Conviction levels are NOT enforced (Trust/Skeptic may improve wording
    while keeping the same %, but small rounding differences are ignored).
    """
    orig_stance = original.get("overall_stance", "").upper().strip()
    opt_stance  = optimized.get("overall_stance", "").upper().strip()

    if orig_stance not in VALID_STANCES or opt_stance not in VALID_STANCES:
        logger.warning(f"core_compromised: invalid stance(s) orig={orig_stance} opt={opt_stance}")
        return True

    if orig_stance != opt_stance:
        logger.warning(f"core_compromised: stance changed {orig_stance} → {opt_stance}")
        return True

    orig_rec = original.get("investment_recommendations", {})
    opt_rec  = optimized.get("investment_recommendations", {})

    for horizon in ("one_day", "one_week", "one_month"):
        orig_pos = orig_rec.get(horizon, {}).get("position", "").upper().strip()
        opt_pos  = opt_rec.get(horizon,  {}).get("position", "").upper().strip()

        if orig_pos not in VALID_POSITIONS:
            logger.warning(f"core_compromised: original {horizon}.position invalid: {orig_pos!r}")
            return True
        if opt_pos not in VALID_POSITIONS:
            logger.warning(f"core_compromised: optimized {horizon}.position invalid: {opt_pos!r}")
            return True
        if orig_pos != opt_pos:
            logger.warning(f"core_compromised: {horizon} position changed {orig_pos} → {opt_pos}")
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 1: Safe Collaborative Debate  (paper Section 2.3.2)
# ─────────────────────────────────────────────────────────────────────────────

def run_safe_debate(
    synthesis: dict,
    p3_data: Optional[dict],
    p4_data: dict,
    trust_client,
    skeptic_client,
    leader_client,
) -> tuple[dict, dict]:
    """
    Implements Algorithm 1 from the paper.

    Input:
        R0  = synthesis      (original report as dict)
        A   = p3_data + p4_data  (agent analysis context)

    Output:
        (R*, debate_log)
        R*        — optimized report dict (or R0 if safety fails)
        debate_log — step-by-step trace for audit

    Algorithm:
        1. Safety Check: Validate R0 structure
        2. if ¬has_recommendations(R0): return R0
        3. Trust Phase:   R1 ← optimize(R0, A)   — preserve + strengthen evidence
        4. Skeptic Phase: R2 ← review(R1, A)      — add risk / hedge strategies
        5. Leader Phase:  R* ← synthesize(R2, A)  — final optimized report
        6. Final Check: Validate R* integrity
        7. if core_compromised(R*, R0): return R0
        8. return R*, L
    """
    from debate.src.prompts import (
        TRUST_SYSTEM_PROMPT, SKEPTIC_SYSTEM_PROMPT, LEADER_SYSTEM_PROMPT,
        build_trust_prompt, build_skeptic_prompt, build_leader_prompt,
    )
    from debate.src.llm_client import safe_parse_json

    debate_log = {
        "source_file":    synthesis.get("source_file", "unknown"),
        "original_stance": synthesis.get("overall_stance", "N/A"),
        "steps":          [],
        "final_source":   None,   # "optimized" or "fallback_R0"
    }

    # ── Step 1-2: Safety Check on R0 ─────────────────────────────────────────
    if not has_recommendations(synthesis):
        logger.info("SAFETY: R0 lacks valid recommendations — returning R0 unchanged.")
        debate_log["steps"].append({
            "step": "safety_check_R0",
            "result": "FAILED — no valid recommendations found",
        })
        debate_log["final_source"] = "fallback_R0"
        return synthesis, debate_log

    debate_log["steps"].append({
        "step": "safety_check_R0",
        "result": "PASSED",
    })

    # ── Step 3: Trust Phase ───────────────────────────────────────────────────
    logger.info("Running Trust Agent...")
    trust_user = build_trust_prompt(synthesis, p3_data, p4_data)
    trust_response = trust_client.chat(TRUST_SYSTEM_PROMPT, trust_user)
    debate_log["steps"].append({
        "step":   "trust_phase",
        "result": "completed",
        "length": len(trust_response),
    })
    logger.info(f"Trust Agent response: {len(trust_response)} chars")

    # ── Step 4: Skeptic Phase ─────────────────────────────────────────────────
    logger.info("Running Skeptic Agent...")
    skeptic_user = build_skeptic_prompt(synthesis, trust_response, p3_data, p4_data)
    skeptic_response = skeptic_client.chat(SKEPTIC_SYSTEM_PROMPT, skeptic_user)
    debate_log["steps"].append({
        "step":   "skeptic_phase",
        "result": "completed",
        "length": len(skeptic_response),
    })
    logger.info(f"Skeptic Agent response: {len(skeptic_response)} chars")

    # ── Step 5: Leader Phase ──────────────────────────────────────────────────
    logger.info("Running Leader Agent...")
    leader_user = build_leader_prompt(synthesis, trust_response, skeptic_response)
    leader_response = leader_client.chat(LEADER_SYSTEM_PROMPT, leader_user)
    debate_log["steps"].append({
        "step":   "leader_phase",
        "result": "completed",
        "length": len(leader_response),
    })
    logger.info(f"Leader Agent response: {len(leader_response)} chars")

    # ── Step 6: Parse R* ──────────────────────────────────────────────────────
    optimized = safe_parse_json(leader_response)
    if optimized is None:
        logger.warning("Leader Agent returned non-parseable JSON — falling back to R0.")
        debate_log["steps"].append({
            "step": "parse_R*",
            "result": "FAILED — JSON parse error; storing raw text",
        })
        debate_log["final_source"] = "fallback_R0"
        # Still store the raw text for inspection
        fallback = dict(synthesis)
        fallback["_debate_raw_leader"] = leader_response
        fallback["_debate_log"] = debate_log
        return fallback, debate_log

    # Carry forward source_file so downstream scripts can match files
    if "source_file" not in optimized:
        optimized["source_file"] = synthesis.get("source_file", "unknown")

    debate_log["steps"].append({
        "step":   "parse_R*",
        "result": "PASSED",
    })

    # ── Step 7: Final Safety Check on R* ─────────────────────────────────────
    if core_compromised(optimized, synthesis):
        logger.warning("SAFETY: core_compromised — returning R0 unchanged.")
        debate_log["steps"].append({
            "step":   "safety_check_R*",
            "result": "FAILED — core recommendations changed; reverting to R0",
        })
        debate_log["final_source"] = "fallback_R0"
        # Store the optimized output for analysis even though we revert
        synthesis["_debate_compromised_output"] = optimized
        synthesis["_debate_log"] = debate_log
        return synthesis, debate_log

    debate_log["steps"].append({
        "step":   "safety_check_R*",
        "result": "PASSED",
    })
    debate_log["final_source"] = "optimized"
    optimized["_debate_log"] = debate_log
    return optimized, debate_log
