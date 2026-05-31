# FinDebate — Person 5: Safe Collaborative Debate Mechanism

Implements **Section 2.3 + Algorithm 1** of the FinDebate paper.

## LLM Provider: Gemini 2.5 Flash (FREE)
- Best freely available model — also used as a benchmark in the paper (Table 1)
- Free tier: **1,500 requests/day**, no credit card needed
- Get your key in 30 seconds: https://aistudio.google.com/apikey

---

## File structure

```
p5_debate/
├── configs/
│   └── config.py          ← EDIT: paths + GEMINI_API_KEY env var name
├── src/
│   ├── prompts.py         ← Agent system prompts (verbatim Appendix D) + prose builders
│   ├── llm_client.py      ← Gemini / OpenAI / Anthropic unified client with retries
│   └── algorithm1.py      ← Algorithm 1: Trust→Skeptic→Leader + safety checks
├── run_debate.py          ← Process ONE file (called by each SLURM task)
├── run_batch.py           ← Process ALL files sequentially (local machine)
├── generate_slurm.py      ← Generate SLURM job array for HPC
├── collect_results.py     ← Validate outputs + print summary table
├── dry_run_test.py        ← Test full pipeline with mock LLM (zero API cost)
└── requirements.txt
```

---

## Quick start

### 1. Install
```bash
pip install google-generativeai
```

### 2. Get your FREE Gemini API key
Go to https://aistudio.google.com/apikey → "Create API key" → copy it.

### 3. Set the key
```bash
export GEMINI_API_KEY="AIza..."
# Add this line to ~/.bashrc so it persists on the cluster
```

### 4. Set paths in config.py
```python
# These match the project structure on your supercomputer:
P4_OUTPUT_DIR = "findebate/outputs"      # Person 4's output folder
P3_OUTPUT_DIR = "findebate/p3_outputs"   # Person 3's output folder
P5_OUTPUT_DIR = "findebate/p5_outputs"   # Person 5 writes here (auto-created)
```

### 5. Dry-run test (no API calls, no cost)
```bash
python dry_run_test.py
# Expected: ALL TESTS PASSED ✓
```

### 6a. Local run (sequential, ~3 hours for 64 files)
```bash
python run_batch.py
# Test first with 3 files:
python run_batch.py --max_files 3
```

### 6b. Supercomputer — RECOMMENDED (~20 min for all 64)
```bash
# On the login node:
python generate_slurm.py \
    --project_dir $HOME/findebate/p5_debate \
    --p4_dir findebate/outputs \
    --p3_dir findebate/p3_outputs \
    --out_dir findebate/p5_outputs \
    --conda_env findebate        # your conda env name

sbatch slurm/job_array.sh
squeue -u $USER                  # monitor jobs
```

### 7. Check results after all jobs finish
```bash
python collect_results.py
# Prints a table: source_file | orig_stance | final_stance | 1D | 1W | 1M | SAFETY
# Writes findebate/p5_outputs/final_summary.json
# Lists any missing files with re-run commands
```

---

## Output format per file

`findebate/p5_outputs/ABM_q3_2021_p5_output.json`:
```json
{
  "source_file": "ABM_q3_2021",
  "final_source": "optimized",
  "debate_result": {
    "overall_stance": "BULLISH",
    "overall_conviction": "75%",
    "executive_summary": "...",
    "investment_recommendations": {
      "one_day":   {"position": "LONG",  "conviction": "75%", "rationale": "..."},
      "one_week":  {"position": "LONG",  "conviction": "75%", "rationale": "..."},
      "one_month": {"position": "LONG",  "conviction": "75%", "rationale": "..."}
    },
    "trust_enhancements": "...",
    "skeptic_risk_additions": "...",
    "risk_reward": { "upside_catalysts": [...], "downside_risks": [...], ... },
    "investment_conclusion": { "final_stance": "BULLISH", "top_3_insights": [...] },
    "debate_log": { "trust_summary": "...", "skeptic_summary": "...", ... }
  },
  "summary": {
    "original_stance": "BULLISH", "final_stance": "BULLISH",
    "final_1day": "LONG", "final_1week": "LONG", "final_1month": "LONG",
    "safety_passed": true
  }
}
```

---

## Algorithm 1 safety guarantees (paper Section 2.3.2)

```
Input:  R0 (original synthesis), A (agent analysis context)
Output: R* (optimized report) or R0 (if safety fails)

1. has_recommendations(R0)?  → if NO:  return R0 unchanged
2. Trust Agent:  R1 = strengthen_evidence(R0, A)
3. Skeptic Agent: R2 = add_risks(R1, A)
4. Leader Agent: R* = synthesize(R2, A)  [outputs JSON]
5. core_compromised(R*, R0)? → if YES: return R0 unchanged
6. return R*
```

`core_compromised` checks that **overall_stance** and all three **position** fields are **identical** to R0. The debate can only strengthen — never redirect.

---

## Rate limits (Gemini free tier)

| Limit | Value | Our usage (5 concurrent tasks) |
|---|---|---|
| Requests/minute | 15 RPM | 5 tasks × 3 calls = 15 RPM ✓ |
| Requests/day | 1,500 | 64 files × 3 calls = 192 total ✓ |
| Tokens/minute | 1M | Well within limits ✓ |
