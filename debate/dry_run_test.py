#!/usr/bin/env python3
"""
FinDebate — Person 5
dry_run_test.py

Tests the full debate pipeline with a MOCK LLM (no API calls).
Run this to verify everything works before using real API keys.

Usage:
    cd /home/claude/p5_debate
    python dry_run_test.py
"""

import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.prompts   import (TRUST_SYSTEM_PROMPT, SKEPTIC_SYSTEM_PROMPT, LEADER_SYSTEM_PROMPT,
                            build_trust_prompt, build_skeptic_prompt, build_leader_prompt)
from src.algorithm1 import has_recommendations, core_compromised, run_safe_debate
from src.llm_client import safe_parse_json


# ── Mock LLM client ────────────────────────────────────────────────────────────
class MockClient:
    """Returns realistic-looking but fake LLM responses."""

    def __init__(self, role: str):
        self.role = role

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.role == "leader":
            # Must return valid JSON matching the schema
            return json.dumps({
                "overall_stance": "BULLISH",
                "overall_conviction": "75%",
                "executive_summary": "Mock enhanced executive summary from Leader Agent.",
                "investment_recommendations": {
                    "one_day":   {"position": "LONG",  "conviction": "75%", "rationale": "Mock 1-day rationale."},
                    "one_week":  {"position": "LONG",  "conviction": "75%", "rationale": "Mock 1-week rationale."},
                    "one_month": {"position": "LONG",  "conviction": "75%", "rationale": "Mock 1-month rationale."},
                },
                "trust_enhancements": "Added evidence about strong earnings guidance and revenue momentum.",
                "skeptic_risk_additions": "Identified macro headwinds and interest rate sensitivity risks.",
                "risk_reward": {
                    "upside_catalysts": ["Strong Q4 guidance beat", "Revenue momentum"],
                    "downside_risks":   ["Macro uncertainty", "Data scarcity"],
                    "position_sizing":  "0.5% of portfolio",
                    "hedge_strategies": ["Index futures", "Stop-loss at -5%"],
                },
                "investment_conclusion": {
                    "final_stance": "BULLISH",
                    "conviction":   "75%",
                    "top_3_insights": [
                        "Guidance raise signals management confidence.",
                        "Moderate data risk tempered by qualitative positives.",
                        "Tactical long with conservative sizing recommended.",
                    ],
                },
                "debate_log": {
                    "trust_summary":   "Strengthened evidence for guidance raise impact.",
                    "skeptic_summary": "Added macro risk and hedging language.",
                    "synthesis_note":  "Balanced conviction with enhanced risk disclosure.",
                },
            })
        else:
            return f"Mock {self.role} agent prose response for testing purposes. " \
                   f"Recommendations preserved: LONG LONG LONG."


# ── Load a real P4 file for testing ───────────────────────────────────────────
P4_FILE = "/tmp/outputs/ABM_q3_2021_p4_output.json"
P3_FILE = "/tmp/p3_outputs/ABM_q3_2021_p3_output.json"

def test_safety_checks():
    print("\n── Test: Safety Checks ──────────────────────────────────")
    with open(P4_FILE) as f:
        p4 = json.load(f)
    synthesis = p4["agents"]["synthesis"]
    synthesis["source_file"] = "ABM_q3_2021"

    # Should PASS
    assert has_recommendations(synthesis), "FAIL: should pass"
    print("  has_recommendations(valid synthesis)  ✓")

    # Should FAIL on broken data
    broken = {"overall_stance": "MAYBE", "investment_recommendations": {}}
    assert not has_recommendations(broken), "FAIL: should fail"
    print("  has_recommendations(broken synthesis) ✓")

    # core_compromised: same → not compromised
    assert not core_compromised(synthesis, synthesis), "FAIL: identical should not be compromised"
    print("  core_compromised(same, same)          ✓")

    # core_compromised: flipped stance → compromised
    flipped = dict(synthesis)
    flipped["overall_stance"] = "BEARISH"
    assert core_compromised(flipped, synthesis), "FAIL: flipped stance should be compromised"
    print("  core_compromised(flipped stance)      ✓")
    print("  ALL SAFETY CHECKS PASSED ✓")


def test_prompt_builders():
    print("\n── Test: Prompt Builders ────────────────────────────────")
    with open(P4_FILE) as f:
        p4 = json.load(f)
    with open(P3_FILE) as f:
        p3 = json.load(f)
    synthesis = p4["agents"]["synthesis"]
    synthesis["source_file"] = "ABM_q3_2021"

    trust_prompt = build_trust_prompt(synthesis, p3, p4)
    assert "LONG" in trust_prompt or "SHORT" in trust_prompt or "NEUTRAL" in trust_prompt
    assert len(trust_prompt) > 500
    print(f"  build_trust_prompt   ✓ ({len(trust_prompt)} chars)")

    skeptic_prompt = build_skeptic_prompt(synthesis, "mock trust response", p3, p4)
    assert len(skeptic_prompt) > 500
    print(f"  build_skeptic_prompt ✓ ({len(skeptic_prompt)} chars)")

    leader_prompt = build_leader_prompt(synthesis, "mock trust", "mock skeptic")
    assert "JSON" in leader_prompt
    assert len(leader_prompt) > 500
    print(f"  build_leader_prompt  ✓ ({len(leader_prompt)} chars)")


def test_full_pipeline():
    print("\n── Test: Full Algorithm 1 Pipeline (Mock) ───────────────")
    with open(P4_FILE) as f:
        p4 = json.load(f)
    with open(P3_FILE) as f:
        p3 = json.load(f)
    synthesis = p4["agents"]["synthesis"]
    synthesis["source_file"] = "ABM_q3_2021"

    optimized, log = run_safe_debate(
        synthesis      = synthesis,
        p3_data        = p3,
        p4_data        = p4,
        trust_client   = MockClient("trust"),
        skeptic_client = MockClient("skeptic"),
        leader_client  = MockClient("leader"),
    )

    print(f"  Final source  : {log['final_source']}")
    print(f"  Debate steps  : {[s['step'] for s in log['steps']]}")
    print(f"  Final stance  : {optimized.get('overall_stance')}")
    print(f"  1-day         : {optimized.get('investment_recommendations',{}).get('one_day',{}).get('position')}")

    assert log["final_source"] == "optimized", f"Expected 'optimized', got {log['final_source']}"
    assert optimized["overall_stance"] == synthesis["overall_stance"], "Stance changed!"
    print("  FULL PIPELINE PASSED ✓")


def test_json_parser():
    print("\n── Test: JSON Parser ────────────────────────────────────")
    # With fences
    fenced = '```json\n{"key": "value"}\n```'
    r = safe_parse_json(fenced)
    assert r == {"key": "value"}, f"Failed fenced: {r}"
    print("  safe_parse_json(fenced)   ✓")

    # Clean
    r2 = safe_parse_json('{"a": 1}')
    assert r2 == {"a": 1}
    print("  safe_parse_json(clean)    ✓")

    # Garbage
    r3 = safe_parse_json("not json at all")
    assert r3 is None
    print("  safe_parse_json(garbage)  ✓ → None")


if __name__ == "__main__":
    print("=" * 55)
    print("FinDebate P5 — Dry Run Test")
    print("=" * 55)
    test_safety_checks()
    test_prompt_builders()
    test_json_parser()
    test_full_pipeline()
    print("\n" + "="*55)
    print("ALL TESTS PASSED ✓  — ready for real API calls")
    print("="*55)
