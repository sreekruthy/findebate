import asyncio
import json
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from pipeline import PIPELINE_AVAILABLE, run_live_pipeline
from precomputed import get_available_companies, load_precomputed


load_dotenv()

app = FastAPI(title="FinDebate Portal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}


class AnalyzeRequest(BaseModel):
    question: str = ""
    ticker: str | None = None
    mode: str = "precomputed"


@app.get("/api/health")
async def health():
    return {"ok": True, "pipeline_available": PIPELINE_AVAILABLE}


@app.get("/api/companies")
async def companies():
    return get_available_companies()


@app.get("/api/config")
async def config():
    return {"pipeline_available": PIPELINE_AVAILABLE}


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    if request.mode not in {"precomputed", "live"}:
        raise HTTPException(status_code=400, detail="mode must be precomputed or live")
    if request.mode == "precomputed" and not request.ticker:
        raise HTTPException(status_code=400, detail="ticker is required for precomputed mode")
    if request.mode == "live" and not request.question.strip():
        raise HTTPException(status_code=400, detail="question is required for live mode")

    job_id = str(uuid4())
    ticker = (request.ticker or "").upper() or None
    jobs[job_id] = {
        "status": "pending",
        "events": [],
        "question": request.question,
        "mode": request.mode,
        "result": None,
    }
    if request.mode == "precomputed":
        asyncio.create_task(stream_precomputed(job_id, ticker))
    else:
        asyncio.create_task(run_live_pipeline(job_id, request.question, ticker or "", jobs))
    return {"job_id": job_id, "ticker": ticker, "mode": request.mode}


@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_generator():
        cursor = 0
        while True:
            events = jobs[job_id]["events"]
            while cursor < len(events):
                yield {"data": json.dumps(events[cursor])}
                cursor += 1
            if jobs[job_id]["status"] in {"complete", "error"} and cursor >= len(events):
                break
            yield {"event": "heartbeat", "data": json.dumps({"status": jobs[job_id]["status"]})}
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@app.get("/api/results/{job_id}")
async def results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job not found")
    if jobs[job_id]["result"] is None:
        raise HTTPException(status_code=202, detail="result not ready")
    return jobs[job_id]["result"]


async def stream_precomputed(job_id: str, ticker: str):
    result = load_precomputed(ticker)
    if not result:
        jobs[job_id]["events"].append({"stage": "error", "status": "failed", "data": {"message": "Ticker not found"}})
        jobs[job_id]["status"] = "error"
        return

    jobs[job_id]["status"] = "streaming"
    stages = [
        ("rag", 1.2, {"n_chunks": 8, "top_sources": [result["source_file"]], "preview": result.get("rag_preview", "")}),
        ("agents", 0.8, None),
        ("synthesis", 1.4, {"stance": result["stance"], "preview": result["executive_summary"][:320]}),
        ("debate", 1.2, {"conviction": result["conviction"]}),
        ("judge", 0.9, {"scores": result["scores"]}),
    ]
    for stage, delay, data in stages:
        if stage == "agents":
            for agent in ["earnings", "market", "sentiment", "valuation", "risk"]:
                await asyncio.sleep(0.55)
                jobs[job_id]["events"].append(
                    {
                        "stage": "agents",
                        "status": "done",
                        "agent": agent,
                        "data": {"summary": result["agent_outputs"].get(agent, "")[:220]},
                    }
                )
            jobs[job_id]["events"].append({"stage": "agents", "status": "all_done", "data": {}})
        else:
            jobs[job_id]["events"].append({"stage": stage, "status": "running", "data": {}})
            await asyncio.sleep(delay)
            jobs[job_id]["events"].append({"stage": stage, "status": "done", "data": data or {}})
    await asyncio.sleep(0.4)
    jobs[job_id]["events"].append({"stage": "complete", "status": "done", "data": result})
    jobs[job_id]["status"] = "complete"
    jobs[job_id]["result"] = result
