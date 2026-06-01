import asyncio

from agents_wrapper import (
    PIPELINE_AVAILABLE as AGENTS_AVAILABLE,
    run_earnings,
    run_market,
    run_p4_agents,
    run_sentiment_from_chunks,
)
from debate_wrapper import PIPELINE_AVAILABLE as DEBATE_AVAILABLE, run_debate
from judge_wrapper import build_report_text, score_report
from rag_wrapper import chunks_to_text, get_agent_context, infer_source_file, merge_chunks, retrieve


PIPELINE_AVAILABLE = AGENTS_AVAILABLE and DEBATE_AVAILABLE


async def run_live_pipeline(job_id: str, question: str, ticker: str, jobs: dict):
    def emit(stage, status, data=None, agent=None):
        event = {"stage": stage, "status": status, "data": data or {}}
        if agent:
            event["agent"] = agent
        jobs[job_id]["events"].append(event)

    try:
        if not PIPELINE_AVAILABLE:
            raise RuntimeError("Live pipeline is unavailable. Check ChromaDB, agent code folders, and API keys.")

        loop = asyncio.get_event_loop()
        emit("rag", "running")
        source_file = infer_source_file(question, ticker)
        top_chunks = retrieve(question, top_k=6, source_file_filter=source_file or None)
        if not source_file:
            source_file = infer_source_file(question)
        if not source_file and top_chunks:
            source_file = top_chunks[0]["source_file"]
            top_chunks = retrieve(question, top_k=6, source_file_filter=source_file)
        if source_file and not top_chunks:
            top_chunks = retrieve(source_file, top_k=6, source_file_filter=source_file)
        rag_context = get_agent_context(source_file=source_file or None, top_k=3)
        for agent_name, agent_chunks in rag_context.items():
            rag_context[agent_name] = merge_chunks(top_chunks, agent_chunks, limit=5)
        emit(
            "rag",
            "done",
            {
                "n_chunks": len(top_chunks),
                "top_sources": list({chunk["source_file"] for chunk in top_chunks[:3]}),
                "matched_source": source_file,
                "retrieval": top_chunks[0].get("retrieval", "none") if top_chunks else "none",
                "preview": top_chunks[0]["chunk"][:220] if top_chunks else "",
            },
        )

        emit("agents", "running")

        async def run_one_agent(agent_name, fn, context_value):
            result = await loop.run_in_executor(None, fn, source_file, context_value)
            key_points = result.get("key_points", [])
            summary = key_points[0] if isinstance(key_points, list) and key_points else result.get("reasoning", "")
            emit("agents", "done", {"summary": str(summary)[:220], "score": result.get("score", 0)}, agent=agent_name)
            return result

        earnings_res, market_res, sentiment_res = await asyncio.gather(
            run_one_agent("earnings", run_earnings, chunks_to_text(rag_context.get("earnings_agent", []))),
            run_one_agent("market", run_market, chunks_to_text(rag_context.get("market_agent", []))),
            run_one_agent("sentiment", run_sentiment_from_chunks, rag_context.get("sentiment_agent", [])),
        )
        emit("agents", "all_done")

        emit("synthesis", "running")
        p3_outputs = {"earnings": earnings_res, "market": market_res, "sentiment": sentiment_res}
        p4_results = await run_p4_agents(source_file, p3_outputs, fallback_chunks=top_chunks)
        valuation_res = p4_results["valuation"]
        risk_res = p4_results["risk"]
        synthesis_res = p4_results["synthesis"]
        emit(
            "synthesis",
            "done",
            {
                "stance": synthesis_res.get("overall_stance", "UNKNOWN"),
                "preview": synthesis_res.get("executive_summary", "")[:320],
                "report_length": len(str(synthesis_res)),
            },
        )

        emit("debate", "running")
        p3_data = {
            "source_file": source_file,
            "agents": {"earnings": earnings_res, "market": market_res, "sentiment": sentiment_res},
        }
        p4_data = {
            "source_file": source_file,
            "agents": {"valuation": valuation_res, "risk": risk_res, "synthesis": synthesis_res},
        }
        optimized, debate_log = await run_debate(synthesis_res, p3_data, p4_data)
        if debate_log:
            optimized["_debate_log"] = debate_log
        emit(
            "debate",
            "done",
            {
                "conviction": optimized.get("overall_conviction", "75%"),
                "trust_additions": len(str(optimized.get("trust_enhancements", ""))),
                "risk_factors": len(str(optimized.get("skeptic_risk_additions", ""))),
                "final_length": len(str(optimized)),
            },
        )

        emit("judge", "running")
        scores = await score_report(build_report_text(optimized))
        emit("judge", "done", {"scores": scores})

        result = {
            "ticker": (ticker or source_file.split("_")[0] or "LIVE").upper(),
            "company": (ticker or source_file.split("_")[0] or "Live Query").upper(),
            "source_file": source_file,
            "stance": optimized.get("overall_stance", "NEUTRAL"),
            "conviction": optimized.get("overall_conviction", "75%"),
            "executive_summary": optimized.get("executive_summary", ""),
            "investment_recommendations": optimized.get("investment_recommendations", {}),
            "trust_text": str(optimized.get("trust_enhancements", ""))[:1200],
            "skeptic_text": str(optimized.get("skeptic_risk_additions", ""))[:1200],
            "risk_reward": str(optimized.get("risk_reward", "")),
            "conclusion": str(optimized.get("investment_conclusion", "")),
            "debate_log": debate_log.get("steps", []) if isinstance(debate_log, dict) else [],
            "scores": scores,
            "agent_outputs": {
                "earnings": str(earnings_res.get("key_points", earnings_res))[:900],
                "market": str(market_res.get("key_points", market_res))[:900],
                "sentiment": str(sentiment_res.get("key_points", sentiment_res))[:900],
                "valuation": str(valuation_res.get("key_points", valuation_res))[:900],
                "risk": str(risk_res.get("key_points", risk_res))[:900],
            },
            "rag_chunks": [chunk["chunk"][:300] for chunk in top_chunks[:5]],
        }
        emit("complete", "done", result)
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = result
    except Exception as exc:
        emit("error", "failed", {"message": str(exc)})
        jobs[job_id]["status"] = "error"
        jobs[job_id]["result"] = {"error": str(exc)}
