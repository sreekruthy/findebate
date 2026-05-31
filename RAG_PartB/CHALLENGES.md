# Challenges, Debugging & Learnings
**P2 — RAG Module Part B | Yajat Kumar**

---

## Overview

This document describes the real challenges encountered during implementation and execution of the FinDebate pipeline on HPC. What began as a straightforward retrieval task evolved into an extended debugging and problem-solving process spanning multiple days, involving API quota management, HPC constraints, code fixes, and pipeline orchestration across a team.

---

## Challenge 1 — Getting P1's ChromaDB

The first challenge was simply accessing P1's ChromaDB. Since ChromaDB is stored on Google Drive, and each Colab session mounts only the user's own Drive, there was no direct way to access a teammate's database. We explored three options:

- Sharing via Google Drive shortcuts — didn't work, Colab can't read shortcuts
- Sharing the folder link — worked, but required downloading and re-uploading
- Re-running P1's ingestion code — would have worked but required the original dataset

The solution was to download the `findebate_chromadb` folder from P1's Drive to MacBook, then upload it to my own Drive. This gave me a working local copy with 6,963 chunks confirmed. A simple problem in hindsight, but it took significant back-and-forth to figure out the right approach.

---

## Challenge 2 — Version Inconsistencies

Once connected, ChromaDB threw a version mismatch warning. P1's code used `chromadb==0.5.23` but the actual installed version on HPC was `0.6.3`. This caused confusion because the handoff documentation stated the wrong version, which P4's teammate's LLM flagged during integration. The lesson: always verify installed versions rather than assuming what the requirements file says.

Similarly, the Google Generative AI package was deprecated mid-project. The old `google.generativeai` package was replaced by `google.genai`, causing FutureWarning messages in every run. While not breaking, it added noise to logs and created confusion about whether outputs were reliable.

---

## Challenge 3 — HPC Login and Setup

Logging into HPC for the first time introduced several small but time-consuming issues:

- The `~` vs `~/` distinction in Linux caused an accidental folder named `~findebate` instead of `~/findebate` — the folder had to be deleted and recreated correctly
- The `module load python/3.10` command in `setup_env.sh` failed because this HPC doesn't use the module system — the line had to be removed
- `sentence-transformers` failed to install the first time due to a known pip resolver bug in older pip versions — fixed by upgrading pip first
- The `Ctrl+C` command in the terminal was misunderstood as potentially cancelling the running HPC job — it only stops the log viewer, not the job itself

Each of these was small individually but cumulatively consumed significant time during setup.

---

## Challenge 4 — Running P4's Pipeline: The API Quota Problem

This was the most significant and time-consuming challenge of the entire project. After completing my P2 tasks, I was asked to run P4's pipeline on HPC since my teammate was out of station and HPC access requires being on the university network.

**First attempt — parallel jobs:**
The initial plan was to run all 64 transcripts simultaneously across 3 SLURM jobs. This failed immediately — the student HPC account has a `AssocGrpCpuLimit` that allows only one job at a time. Jobs 2 and 3 sat in pending state indefinitely.

**Second attempt — sequential with dependencies:**
Switched to SLURM's `--dependency=afterok:JOBID` feature to chain jobs automatically. This worked, but a new problem emerged: only 5-6 transcripts succeeded before hitting Gemini API rate limits. The free tier has a daily quota of approximately 50-100 requests per key. With 3 API calls per transcript, only 4-6 transcripts were processable per key per day.

**The rotation strategy:**
The solution was to use multiple fresh API keys, each handling exactly 4 transcripts (12 API calls — within free tier limits). A `--start_from` and `--end_at` parameter system was implemented to control which transcripts each job processed. Importantly, `--end_at` didn't exist in the original Pipeline.py — it had to be added manually to prevent jobs from overlapping and overwriting each other's outputs.

Over the course of two days, this required:
- Creating and rotating through more than 15 different Gemini API keys
- Submitting over 20 individual SLURM jobs
- Monitoring checkpoints every 20-30 minutes
- Identifying and rerunning 8 transcripts that had old timestamps after prompt fixes

**Prompt fixes discovered mid-run:**
During execution, it was discovered that P4's prompts had hardcoded `"score": 0.0` as an example value — causing Gemini to always output 0.0 for scores. Additionally, almost all outputs showed NEUTRAL stance due to the model defaulting to safe outputs. Both issues were fixed directly on HPC using nano, and all 64 transcripts were rerun with the corrected prompts.

---

## Challenge 5 — GitHub Push Failures

After completing all 64 transcripts, pushing outputs to GitHub via the API returned 422 errors for several files. Initial investigation suggested file size issues, but the files were only 13-15KB — well under GitHub's 1MB limit. Further investigation revealed the files were already on the Person4 branch from a previous run — the 422 error was GitHub rejecting updates where the content hadn't changed. All 64 files were confirmed present on GitHub without any additional action needed.

---

## Key Learnings

**Technical:**
- SLURM `--dependency=afterok:JOBID` is essential for sequential job chaining on shared HPC systems with allocation limits
- Free tier API quota is a daily limit, not just per-minute — slowing down the rate doesn't help once the daily ceiling is hit
- Always add `--end_at` or equivalent range controls when splitting batch jobs across multiple API keys
- Prompt example values (like `"score": 0.0`) are copied literally by LLMs — never use real numbers as examples in output format specifications
- ChromaDB path on HPC must match exactly — `/dgxa_home/` not `/home/`

**Process:**
- Test on a single transcript before submitting a full batch job — saves hours of wasted quota
- Always verify file timestamps after a rerun to confirm fresh outputs were actually written
- SLURM log files (`*.log` and `*.err`) are the most reliable way to diagnose job failures — not the queue status
- On shared academic HPC systems, always check partition limits before planning parallel execution

---

## Time Investment

The P2 implementation (multi-level retrieval + SLURM setup) took approximately one full day. Supporting P4's pipeline execution across two days added significant additional effort, involving continuous monitoring, debugging, API key rotation, and prompt fixes. The total active working time across both tasks spanned multiple sessions over several days, with the bulk of effort going into problem diagnosis and resolution rather than the initial implementation itself.
